"""Speed diagnostic for GLM 5.3 Flash on mlx-vlm 0.7.12: where does a token's time go, and where is float32 leaking?

What it measures, on the first K decoder layers (lazy load):
  1. a number-format audit of one decode token: the types going into and out of every attention / MLP / expert
     block, hyper-connection and indexer, and the (input, scales) type pair of every 4-bit multiply. A bf16 input
     against fp16 scales is a mixed pair, which MLX upgrades to float32: four times the cost per multiply (bug 4a).
     This audit is how bugs 3 and 4 were found;
  2. whole-slice decode ms/token with one evaluation per token (as generation does), at a short and a long context,
     with a rough projection to the full 45 layers;
  3. ms per part (attention, feed-forward, hyper-connection collapse/expand), synced, to rank them;
  4. the per-token tail outside the layers: final norm + output layer (fed bf16, and fed float32 as bug 4b did:
     about 120 ms against 2 ms) and the top-p sampler.
The audit hooks are installed before patches.py, so with the fixes on they show the types each block actually
computes in (patches.py converts on the way in and out).

This script is for comparing settings, so it has switches for that on purpose (environment variables):
  FIXES=all (default)  import patches.py (all fixes).
  FIXES=12             bugs 1 and 2 only (load + memory), so you can see the float32 tax of bugs 3 and 4 in the
                       audit and the timings. Bug 2's fix stays on: the long-context prefill needs it.
  SCALES=bf16          convert every fp16 quantisation scale in the kept layers to bf16 before they load, the other way to end the
                       bf16 x fp16 mix. patches.py does the opposite (runs each 4-bit multiply in fp16, so the scales
                       stay exact); this lets you compare the two on speed. bf16 scales lose precision: check drift
                       with requant_check.py SCALES_BF16=1 before using it.
  SAVE_H=path.npy / CMP_H=path.npy   save the prompt's final hidden state (after the final norm, so not comparable
                       with memory_test.py's files), or compare against a saved one.
Memory: only the kept layers' weights load (the first 3 layers 0.34 GiB each, each expert layer 4.2 GiB, plus 1.2 GiB
of embeddings and 1.2 GiB output layer): K=4 peaks at about 12 GiB, K=8 at about 30 GiB.
A guard thread exits if system free memory drops under 30% or MLX passes STOP_GB (default 50).

Run (from this folder):
  ~/venvs/mlx-vlm-glm5/bin/python -I speed_diag.py [K=8] [LONG=4000]     (LONG=0 skips the long-context run)
For timings, quit what else uses the GPU first (Activity Monitor > Window > GPU History).
The model path is the one in patches.py (~/models/GLM-5.3-Flash-mlx-4bit).
"""
import os, re, subprocess, sys, threading, time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.expanduser("~/models/GLM-5.3-Flash-mlx-4bit")  # same path as patches.py
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LONG = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
STOP_GB = float(os.environ.get("STOP_GB", "50"))
FIXES = os.environ.get("FIXES", "all")
SCALES = os.environ.get("SCALES", "")
GiB = 2**30
if FIXES not in ("all", "12"):
    sys.exit("FIXES must be 'all' or '12'")


def guard():
    while True:
        try:
            s = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
            free = int(re.search(r"free percentage: (\d+)%", s).group(1))
            if free < 30 or mx.get_active_memory() / GiB > STOP_GB:
                print(f"GUARD: {free}% free, active {mx.get_active_memory() / GiB:.1f} GiB, exiting", flush=True)
                os._exit(3)
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=guard, daemon=True).start()
mx.set_memory_limit(int(STOP_GB) * GiB)
mx.set_cache_limit(2 * GiB)

import mlx_vlm.models.glm5_next.language as L  # noqa: E402
from mlx_vlm.models.base import create_ssm_mask  # noqa: E402
from mlx_vlm.models.deepseek_v4 import hyper_connection as HC  # noqa: E402
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402

# --- number-format audit hooks (record only while AUDIT is on). Installed before patches.py so they sit inside it.
AUDIT = {"on": False, "rows": []}


def _t(x):
    return str(x.dtype).replace("mlx.core.", "")


def audit(cls, name):
    orig = getattr(cls, name)

    def f(self, *a, **k):
        r = orig(self, *a, **k)
        if AUDIT["on"]:
            ins = [_t(x) for x in a if isinstance(x, mx.array)]
            outs = [_t(x) for x in (r if isinstance(r, tuple) else (r,)) if isinstance(x, mx.array)]
            AUDIT["rows"].append(f"{cls.__name__}.{name}: in {ins} out {outs}")
        return r
    setattr(cls, name, f)


for c, n in ((L.Glm5NextLinearAttention, "__call__"), (L.Glm5NextAttention, "__call__"), (L.Glm5NextMLP, "__call__"),
             (L.Glm5NextMoE, "__call__"), (HC.HyperConnection, "__call__"), (L.Glm5NextIndexer, "__call__")):
    audit(c, n)


def audit_q(cls):  # the (input, scales) pair of every 4-bit multiply
    orig = cls.__call__

    def f(self, x, *a, **k):
        r = orig(self, x, *a, **k)
        if AUDIT["on"]:
            sc = self["scales"].dtype
            mixed = {x.dtype, sc} == {mx.bfloat16, mx.float16}
            AUDIT["rows"].append(f"{cls.__name__}: input {_t(x)} x scales {str(sc).replace('mlx.core.', '')} -> "
                                 f"out {_t(r)}{'  (mixed pair: runs in float32)' if mixed else ''}")
        return r
    cls.__call__ = f


for c in (nn.QuantizedLinear, QuantizedSwitchLinear):
    audit_q(c)

_hx = HC.hc_expand


def hc_expand_a(x, residual, post, comb, **kw):
    r = _hx(x, residual, post, comb, **kw)
    if AUDIT["on"]:
        AUDIT["rows"].append(f"hc_expand: branch out {_t(x)}, residual {_t(residual)} -> {_t(r)}")
    return r


HC.hc_expand = hc_expand_a

# --- the fixes
if FIXES == "all":
    import patches  # noqa: E402,F401  (must come before load())
else:
    import mlx_vlm.models.glm5_next.glm5_next as _g5  # noqa: E402
    # bug 1: point the fused gate_up_proj at gate_proj's bit widths
    _g5.Model.quantization_path_aliases = lambda self, p: (
        (p.removeprefix("language_model.").replace(".gate_up_proj", ".gate_proj"),) if ".gate_up_proj" in p else ())
    # bug 2: hand the sparse-attention kernel bf16 so it doesn't fall back to a huge copy
    _sp = L._sparse_prefill_attention
    L._sparse_prefill_attention = lambda q, k, v, i, s, **kw: _sp(
        q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16), i, s, **kw).astype(q.dtype)
from mlx_vlm import load  # noqa: E402

model, processor = load(MODEL_DIR, lazy=True)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
lm = model.language_model
N_FULL = len(lm.model.layers)
lm.model.layers = lm.model.layers[:K]
mx.eval(lm.model.embed_tokens.parameters(), lm.model.norm.parameters(), lm.lm_head.parameters())
nconv = 0
if SCALES == "bf16":
    for layer in lm.model.layers:
        for _, m in layer.named_modules():
            if hasattr(m, "scales") and isinstance(getattr(m, "scales"), mx.array) and m.scales.dtype == mx.float16:
                m.scales = m.scales.astype(mx.bfloat16)
                if getattr(m, "biases", None) is not None:
                    m.biases = m.biases.astype(mx.bfloat16)
                nconv += 1
print(f"fixes {'all (patches.py)' if FIXES == 'all' else 'bugs 1 and 2 only'}; scales {SCALES or 'as stored'}"
      f"{f' ({nconv} quantised modules converted)' if nconv else ''}", flush=True)
for layer in lm.model.layers:
    mx.eval(layer.parameters())
print(f"K={K} of {N_FULL} layers in, active {mx.get_active_memory() / GiB:.1f} GiB; output layer weight "
      f"{lm.lm_head.weight.dtype} {tuple(lm.lm_head.weight.shape)}", flush=True)

q = ("Which of the following is the primary function of red blood cells?\nOptions:\nA. Fight infection\n"
     "B. Carry oxygen\nC. Clot blood\nD. Produce hormones\nAnswer with the letter of the correct option.")
text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
short_ids = tok.encode(text)
filler = tok.encode(" ".join(["The quick brown fox jumps over the lazy dog."] * (LONG // 10)))[:LONG]


def forward(ids, cache):
    h = lm.model.embed_tokens(mx.array([ids]))
    h = mx.repeat(h[:, :, None], lm.config.hc_mult, axis=2)
    topk = None
    for i, layer in enumerate(lm.model.layers):
        m = create_ssm_mask(h[:, :, 0], cache[i]) if layer.block_type == "linear_attention" else None
        h, topk = layer(h, m, cache[i], topk)
    return h


def prefill(ids):
    cache = lm.make_cache()[:K]
    for s in range(0, len(ids), 512):
        h = forward(ids[s:s + 512], cache)
        mx.eval(h)
    return cache


def decode_ms(cache, steps=20, warm=4):
    ts = []
    for s in range(steps):
        t0 = time.perf_counter()
        h = forward([short_ids[-1]], cache)
        mx.eval(h)
        if s >= warm:
            ts.append(time.perf_counter() - t0)
    return 1000 * sum(ts) / len(ts)


cache = prefill(short_ids)
_c2 = lm.make_cache()[:K]
href = forward(short_ids, _c2).astype(mx.float32); mx.eval(href); del _c2
if os.environ.get("SAVE_H"):
    mx.save(os.environ["SAVE_H"], href)
if os.environ.get("CMP_H"):
    ref = mx.load(os.environ["CMP_H"])
    cos = (mx.sum(href * ref, axis=-1) / (mx.linalg.norm(href, axis=-1) * mx.linalg.norm(ref, axis=-1))).flatten()
    print(f"== prompt hidden state vs saved run: mean |diff| {mx.abs(href - ref).mean().item():.5f}, saved mean |x| "
          f"{mx.abs(ref).mean().item():.5f}, cos min {cos.min().item():.6f} mean {cos.mean().item():.6f}", flush=True)
print(f"prefill {len(short_ids)} tok done; active {mx.get_active_memory() / GiB:.1f} peak {mx.get_peak_memory() / GiB:.1f} GiB", flush=True)

# 1. number-format audit, one decode token
AUDIT["on"] = True
mx.eval(forward([short_ids[-1]], cache))
AUDIT["on"] = False
seen = defaultdict(int)
for r in AUDIT["rows"]:
    seen[r] += 1
print("== number-format audit (one decode token; counts are summed over the kept layers)")
for r, n in seen.items():
    print(f"  {n:3d}x {r}")

# 2. whole-slice decode, one eval per token
ms_short = decode_ms(cache)
print(f"== whole slice decode, context ~{len(short_ids)}: {ms_short:.1f} ms/token for {K} layers = "
      f"{ms_short / K:.2f} ms/layer -> all {N_FULL} layers ~{ms_short / K * N_FULL:.0f} ms "
      f"(~{1000 / (ms_short / K * N_FULL):.1f} tok/s, before the output layer and sampler)", flush=True)

# 3. per part, synced
parts = defaultdict(list)
for s in range(12):
    h = lm.model.embed_tokens(mx.array([[short_ids[-1]]]))
    h = mx.repeat(h[:, :, None], lm.config.hc_mult, axis=2)
    mx.eval(h)
    topk = None
    for i, layer in enumerate(lm.model.layers):
        tag = f"L{i} {layer.block_type[:6]}/{type(layer.mlp).__name__[8:] or 'MLP'}"
        m = create_ssm_mask(h[:, :, 0], cache[i]) if layer.block_type == "linear_attention" else None
        t0 = time.perf_counter()
        coll, post, comb = layer.attn_hc(h); coll = layer.input_layernorm(coll); mx.eval(coll, post, comb)
        t1 = time.perf_counter()
        if layer.is_linear:
            out = layer.self_attn(coll, m, cache[i]); mx.eval(out)
        else:
            out, topk = layer.self_attn(coll, None, cache[i], topk); mx.eval(out, topk)
        t2 = time.perf_counter()
        h = _hx(out, h, post, comb); mx.eval(h)
        t3 = time.perf_counter()
        coll, post, comb = layer.ffn_hc(h); coll = layer.post_attention_layernorm(coll); mx.eval(coll)
        out = layer.mlp(coll); mx.eval(out)
        t4 = time.perf_counter()
        h = _hx(out, h, post, comb); mx.eval(h)
        t5 = time.perf_counter()
        if s >= 3:
            for k_, v_ in (("hc_collapse", t1 - t0), ("attn", t2 - t1), ("hc_expand", t3 - t2 + t5 - t4),
                           ("ffn(+hc)", t4 - t3)):
                parts[(tag, k_)].append(v_)
print("== per part (synced, ms)")
for (tag, k_), v in parts.items():
    print(f"  {tag:22s} {k_:12s} {1000 * sum(v) / len(v):6.2f}")

# 4. the tail: final norm + output layer + sampler
from mlx_vlm.sample_utils import make_sampler  # noqa: E402
sampler = make_sampler(temp=0.7, top_p=0.95)
hid = mx.random.normal((1, 1, lm.config.hidden_size)).astype(mx.bfloat16)
for name, fn in (("norm + output layer, bf16 in", lambda: lm.lm_head(lm.model.norm(hid))),
                 ("norm + output layer, float32 in (bug 4b)", lambda: lm.lm_head(lm.model.norm(hid.astype(mx.float32))))):
    ts = []
    for s in range(10):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    print(f"== {name}: {1000 * sum(ts[2:]) / 8:.2f} ms", flush=True)
logits = lm.lm_head(lm.model.norm(hid))[:, -1]
mx.eval(logits)
try:
    ts = []
    for s in range(10):
        t0 = time.perf_counter(); mx.eval(sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))); ts.append(time.perf_counter() - t0)
    print(f"== top-p sampler: {1000 * sum(ts[2:]) / 8:.2f} ms", flush=True)
except Exception as e:
    print(f"== sampler not timed: {e!r}", flush=True)

# 2b. long context
if LONG:
    cache = prefill(short_ids + filler)
    print(f"prefill {len(short_ids) + len(filler)} tok; active {mx.get_active_memory() / GiB:.1f} peak {mx.get_peak_memory() / GiB:.1f} GiB", flush=True)
    ms_long = decode_ms(cache)
    print(f"== whole slice decode, context ~{len(short_ids) + len(filler)}: {ms_long:.1f} ms/token ({ms_long / K:.2f} ms/layer)", flush=True)
print("SPEED DIAG DONE", flush=True)

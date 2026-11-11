"""Layer-by-layer memory test for GLM 5.3 Flash on mlx-vlm 0.7.12.

What it measures: which part of the model allocates memory on a forward pass. It loads the model LAZILY, keeps only
the first K decoder layers, and sends one short multiple-choice prompt through them one layer at a time, printing
MLX's active and peak memory after each layer's weights load and after its forward pass. For the attention
indexer, the sparse attention and the expert block it also prints how much extra memory each call needed, and for
every sparse-attention call it prints the number types arriving and whether the fast kernel accepted them.

This is how bug 2 was found. On the full model, MLX went from 186 to 226 GiB in two seconds on a ~300-token prompt:
the sparse-attention kernel only takes 16-bit inputs, the inputs arrived as float32, and the fallback made a copy of
~2,000 earlier positions for every prompt token. On this test (K=4, a 91-token prompt) the sparse attention needs an
extra 23.6 GiB without the fix in patches.py, and almost none with it.

Memory: only the kept layers' weights ever load: the embeddings 1.2 GiB, each of the first 3 (dense) layers 0.34 GiB,
each expert (MoE) layer 4.2 GiB. With the fixes, K=4 peaks at about 11 GiB and K=8 at about 30 GiB. A guard thread exits if system free memory drops under 30% or MLX's active memory passes STOP_GB
(default 50).

Run (from this folder):
  ~/venvs/mlx-vlm-glm5/bin/python -I memory_test.py [K=8]
Layer 3 is the first sparse-attention layer and the first expert (MoE) layer, so K=4 is the smallest K that touches
every kind of block.

Switches (environment variables):
  FIXES=1 (default)  import patches.py, so you see the fixed model.
  FIXES=0            (K=4 or less) only bug 1's load fix, so bug 2's memory spike happens on the slice: +23.6 GiB in the first
                     sparse-attention layer at K=4 (about 31 GiB in all), more for each one after it.
                     Keep K small and the guard on; never do this on the full model.
  CHECK=1            (with FIXES=1) also run the unfixed float32 fallback on each sparse-attention call and print how
                     far the bf16 kernel's output is from it. This costs the same memory as FIXES=0.
  WRAP=0             don't print per-call extra memory for the indexer / attention / expert block (automatically off
                     with DECODE_STEPS, since it would skew the timings).
  DECODE_STEPS=N     after the prompt, decode N single tokens through the kept layers and print ms/token per layer
                     (the first 3 steps are warm-up and not counted).
  SAVE_H=path.npy    save the final hidden state (all 4 hyper-connection streams, so not comparable with speed_diag.py's
                     files);  CMP_H=path.npy  compare against a saved one (mean |diff|, cosine).
                     Run once with FIXES=0 and SAVE_H, then with FIXES=1 and CMP_H, to see how far the fixes move the
                     numbers.
The model path is the one in patches.py (~/models/GLM-5.3-Flash-mlx-4bit).
"""
import os, re, subprocess, sys, threading, time

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.expanduser("~/models/GLM-5.3-Flash-mlx-4bit")  # same path as patches.py
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
STOP_GB = float(os.environ.get("STOP_GB", "50"))
FIXES = os.environ.get("FIXES", "1") == "1"
if not FIXES and K > 4:
    sys.exit("FIXES=0 adds ~24 GiB per sparse-attention layer (layers 3, 7, 11, ...): use K=4 or less")
if not FIXES and os.environ.get("CHECK") == "1":
    print("note: CHECK=1 only works with FIXES=1; ignored", flush=True)
GiB = 2**30


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
mx.set_memory_limit(int(STOP_GB) * GiB)  # a guideline for MLX, not a hard stop: the guard is the real limit
mx.set_cache_limit(1 * GiB)

import mlx_vlm.models.glm5_next.language as L  # noqa: E402

# Watch every sparse-attention call. Installed BEFORE patches.py so that, with the fixes, the patch wraps this
# function and the types printed are the ones the kernel actually receives.
_sp = L._sparse_prefill_attention


def sp(q, k, v, idx, scale, **kw):
    ok = L.indexed_sparse_attention(q, k, v, idx, scale) is not None
    print(f"    sparse attention q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)} idx{tuple(idx.shape)}, "
          f"types q {q.dtype} k {k.dtype} v {v.dtype}: fast kernel {'yes' if ok else 'NO (fallback copy)'}", flush=True)
    out = _sp(q, k, v, idx, scale, **kw)
    if FIXES and os.environ.get("CHECK") == "1" and q.dtype == mx.bfloat16:
        f32 = mx.float32
        ref = _sp(q.astype(f32), k.astype(f32), v.astype(f32), idx, scale, **kw)
        d = mx.abs(out.astype(f32) - ref)
        print(f"    bf16 kernel vs float32 fallback: max |diff| {d.max().item():.4f}, mean {d.mean().item():.5f}, "
              f"fallback mean |x| {mx.abs(ref).mean().item():.4f}", flush=True)
    return out


L._sparse_prefill_attention = sp

if FIXES:
    import patches  # noqa: E402,F401  (all the fixes; must come before load())
else:
    # bug 1 only: without it the model won't load at all
    import mlx_vlm.models.glm5_next.glm5_next as _g5  # noqa: E402
    _g5.Model.quantization_path_aliases = lambda self, p: (
        (p.removeprefix("language_model.").replace(".gate_up_proj", ".gate_proj"),) if ".gate_up_proj" in p else ())

from mlx_vlm import load  # noqa: E402
from mlx_vlm.models.base import create_ssm_mask  # noqa: E402


def mem(tag):
    print(f"{tag:42s} active {mx.get_active_memory() / GiB:6.2f}  peak {mx.get_peak_memory() / GiB:6.2f} GiB", flush=True)


print(f"fixes: {'all (patches.py)' if FIXES else 'bug 1 only'}", flush=True)
model, processor = load(MODEL_DIR, lazy=True)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
mem("lazy load")
lm = model.language_model
layers = lm.model.layers
print(f"{len(layers)} layers; keeping the first {K}: " + ", ".join(
    f"{i}:{l.block_type}/{type(l.mlp).__name__}" for i, l in enumerate(layers[:K])), flush=True)
lm.model.layers = layers[:K]
mx.eval(lm.model.embed_tokens.parameters())
mem("embeddings")

# a plain MMLU-style question, thinking on (the chat template's default)
q = ("Which of the following is the primary function of red blood cells?\nOptions:\nA. Fight infection\n"
     "B. Carry oxygen\nC. Clot blood\nD. Produce hormones\nE. Digest fat\nF. Store iron\nG. Make antibodies\n"
     "H. Regulate temperature\nI. Transport glucose only\nJ. Remove carbon dioxide only\n"
     "Answer with the letter of the correct option.")
text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
ids = mx.array(tok.encode(text))[None]
print(f"prompt {ids.shape[1]} tokens", flush=True)


def wrap(cls, name):
    orig = getattr(cls, name)

    def f(self, *a, **k):
        mx.eval(a[0]); mx.reset_peak_memory(); a0 = mx.get_active_memory()
        r = orig(self, *a, **k)
        mx.eval(r)
        print(f"    {cls.__name__}.{name}: peak +{max(0, mx.get_peak_memory() - a0) / GiB:.2f} GiB", flush=True)
        return r
    setattr(cls, name, f)


if os.environ.get("WRAP") != "0" and not int(os.environ.get("DECODE_STEPS", "0")):  # the wraps would skew the timing
    for c, n in ((L.Glm5NextIndexer, "__call__"), (L.Glm5NextAttention, "_attend"), (L.Glm5NextMoE, "__call__")):
        wrap(c, n)

cache = lm.make_cache()
h = lm.model.embed_tokens(ids)
h = mx.repeat(h[:, :, None], lm.config.hc_mult, axis=2)  # the four hyper-connection streams
mx.eval(h)
topk = None
for i, layer in enumerate(lm.model.layers):
    n = sum(v.nbytes for _, v in tree_flatten(layer.parameters())) / GiB
    mx.eval(layer.parameters())
    mem(f"L{i} weights in ({n:.2f} GiB)")
    mx.reset_peak_memory()
    m = create_ssm_mask(h[:, :, 0], cache[i]) if layer.block_type == "linear_attention" else None
    h, topk = layer(h, m, cache[i], topk)
    mx.eval(h, topk) if topk is not None else mx.eval(h)
    mem(f"L{i} forward {layer.block_type}/{type(layer.mlp).__name__}")
print(f"residual stream type after the prompt: {h.dtype}", flush=True)
if os.environ.get("SAVE_H"):
    mx.save(os.environ["SAVE_H"], h.astype(mx.float32))
if os.environ.get("CMP_H"):
    ref = mx.load(os.environ["CMP_H"])
    hf = h.astype(mx.float32)
    d = mx.abs(hf - ref)
    print(f"hidden state vs saved run: mean |diff| {d.mean().item():.5f}, saved mean |x| {mx.abs(ref).mean().item():.5f}, "
          f"cos {(mx.sum(hf * ref) / (mx.linalg.norm(hf) * mx.linalg.norm(ref))).item():.6f}", flush=True)

# decode timing: one token at a time through the kept layers, as generation does
STEPS = int(os.environ.get("DECODE_STEPS", "0"))
if STEPS:
    from collections import defaultdict  # noqa: E402
    times = defaultdict(list)
    tokid = int(ids[0, -1].item())
    for step in range(STEPS):
        x = lm.model.embed_tokens(mx.array([[tokid]]))
        x = mx.repeat(x[:, :, None], lm.config.hc_mult, axis=2)
        mx.eval(x)
        topk = None
        for i, layer in enumerate(lm.model.layers):
            t0 = time.perf_counter()
            m = create_ssm_mask(x[:, :, 0], cache[i]) if layer.block_type == "linear_attention" else None
            x, topk = layer(x, m, cache[i], topk)
            mx.eval(x, topk) if topk is not None else mx.eval(x)
            if step >= 3:
                times[i].append(time.perf_counter() - t0)
    for i, layer in enumerate(lm.model.layers):
        t = times[i]
        if t:
            print(f"decode L{i} {layer.block_type}/{type(layer.mlp).__name__}: {1000 * sum(t) / len(t):.1f} ms/token", flush=True)
    print(f"decode residual stream type: {x.dtype}", flush=True)
print("MEMORY TEST DONE", flush=True)

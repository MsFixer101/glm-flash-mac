"""Drift check: how far does storing the attention and the output layer at fewer bits move the model?

Background: the 4-bit conversion left the projections of the 34 linear-attention layers (qkv_proj and o_proj, about
8.5 GiB) and the output layer (lm_head, about 1.2 GiB) at full 16-bit size, and they're read on every token.
Quantising them in memory after load (speed.py does this; the files on disk are untouched) makes decoding faster,
but it changes the numbers. This script measures by how much, on the first K layers.

What it does: runs the first K layers on a fixed 54-token prompt and keeps the final hidden state and the output
scores (logits); then quantises as the switches below ask, runs again, and compares:
  hidden cos   cosine similarity of the final hidden state per token (1.0 = identical)
  top-1 same   how often the most likely next token is unchanged
  KL           how far the next-token distribution moved (0 = not at all)
Measured (K=8): 8-bit attention: cos 0.99989, top-1 100%, KL 0.0004 (0.0005 with the output layer at 8-bit too,
the speed.py setting). 4-bit attention: cos 0.9975, top-1 92.6%,
KL 0.008. Output layer alone at 8-bit: KL 0.0001. Six experts per token instead of 8 (TOP_K=6): cos 0.99709,
top-1 96.3%, KL 0.0088, as much drift as 4-bit attention for about 4% more speed.

Memory: about 29 GiB peak at K=8 (lazy load). A guard thread exits if system free memory drops under 30% or MLX passes
STOP_GB (default 50).

Run (from this folder):
  QUANT_DENSE=8 QUANT_LM_HEAD=8 ~/venvs/mlx-vlm-glm5/bin/python -I requant_check.py [K=8]
Switches (environment variables; any combination, at least one):
  QUANT_DENSE=8 | 4  linear-attention qkv_proj and o_proj at that many bits (group size 64).
  QUANT_LM_HEAD=8    the output layer at 8-bit. QUANT_DENSE=8 QUANT_LM_HEAD=8 is exactly speed.shrink_attention().
  TOP_K=6            route each token to 6 of its experts instead of 8.
  SCALES_BF16=1      store every fp16 quantisation scale as bf16 instead, so patches.py's fp16 round trip (bug 4a)
                     no longer fires. bf16 keeps 8 bits of precision against fp16's 11: an experiment in how much
                     that loses, not a recommended setting.
The model path is the one in patches.py (~/models/GLM-5.3-Flash-mlx-4bit).
"""
import os, re, subprocess, sys, threading, time

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
STOP_GB = float(os.environ.get("STOP_GB", "50"))
GiB = 2**30


def guard():
    while True:
        try:
            s = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
            if int(re.search(r"free percentage: (\d+)%", s).group(1)) < 30 or mx.get_active_memory() / GiB > STOP_GB:
                print("GUARD exit", flush=True); os._exit(3)
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=guard, daemon=True).start()
mx.set_memory_limit(int(STOP_GB) * GiB)
import patches  # noqa: E402  (must come before load())
import mlx_vlm.models.glm5_next.language as L  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402


def requant(model):
    """QUANT_DENSE / QUANT_LM_HEAD: quantise the dense linear-attention projections and/or the output layer."""
    bits, head = os.environ.get("QUANT_DENSE"), os.environ.get("QUANT_LM_HEAD")
    lm = model.language_model
    done = []

    def pick(path, m):
        if not bits or not isinstance(m, nn.Linear) or isinstance(m, nn.QuantizedLinear) or ".self_attn." not in path:
            return False
        if path.rsplit(".", 1)[-1] not in ("qkv_proj", "o_proj"):
            return False
        if lm.config.layer_types[int(path.split("layers.")[1].split(".")[0])] != "linear_attention":
            return False
        done.append(path)
        return {"group_size": 64, "bits": int(bits)}

    nn.quantize(lm.model, class_predicate=pick)
    if head and not isinstance(lm.lm_head, nn.QuantizedLinear):
        lm.lm_head = nn.QuantizedLinear.from_linear(lm.lm_head, group_size=64, bits=int(head))
        done.append("lm_head")
    return done


def scales_to_bf16(model):
    n = 0
    for _, m in model.language_model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, QuantizedSwitchLinear)) and m["scales"].dtype == mx.float16:
            m.scales = m.scales.astype(mx.bfloat16)
            if getattr(m, "biases", None) is not None:
                m.biases = m.biases.astype(mx.bfloat16)
            n += 1
    return n


def set_top_k(model, k):
    n = 0
    for _, m in model.language_model.named_modules():
        if isinstance(m, L.MoEGate):
            m.top_k = k
            n += 1
    return n


if not (any(os.environ.get(v) for v in ("QUANT_DENSE", "QUANT_LM_HEAD", "TOP_K")) or os.environ.get("SCALES_BF16") == "1"):
    sys.exit("nothing to compare: set QUANT_DENSE, QUANT_LM_HEAD, TOP_K or SCALES_BF16 (see the docstring)")

model, processor = load(patches.MODEL_DIR, lazy=True)
lm = model.language_model
lm.model.layers = lm.model.layers[:K]
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
text = ("The Krebs cycle takes place in the mitochondrial matrix. A court must decide whether a contract formed "
        "under duress can be enforced, and the defendant argues that the plaintiff's promise was illusory. "
        "Compute the derivative of x^3 sin(x) and explain each step.")
ids = mx.array([tok.encode(text)])


def run():
    h = lm.model(ids)
    logits = lm.lm_head(h.astype(mx.bfloat16))
    mx.eval(h, logits)
    return h.astype(mx.float32), logits.astype(mx.float32)


h0, l0 = run()
done = requant(model)
if os.environ.get("SCALES_BF16") == "1":
    done += ["scales_bf16"] * scales_to_bf16(model)
if os.environ.get("TOP_K"):
    done += ["top_k"] * set_top_k(model, int(os.environ["TOP_K"]))
mx.eval(lm.parameters())
h1, l1 = run()


def cos(a, b):
    return (a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1))


c = cos(h0[0], h1[0])
top0, top1 = mx.argmax(l0[0], -1), mx.argmax(l1[0], -1)
p0, p1 = mx.softmax(l0[0], -1), mx.softmax(l1[0], -1)
kl = (p0 * (mx.log(p0 + 1e-12) - mx.log(p1 + 1e-12))).sum(-1)
mx.eval(c, top0, top1, kl)
print(f"changed {len(done)} modules (QUANT_DENSE={os.environ.get('QUANT_DENSE')}, QUANT_LM_HEAD={os.environ.get('QUANT_LM_HEAD')}, "
      f"TOP_K={os.environ.get('TOP_K')}, SCALES_BF16={os.environ.get('SCALES_BF16')}), K={K}, {ids.shape[1]} tokens", flush=True)
print(f"hidden cos mean {c.mean().item():.6f} min {c.min().item():.6f}; top-1 same {(top0 == top1).mean().item() * 100:.1f}%; "
      f"KL mean {kl.mean().item():.5f} max {kl.max().item():.5f}; MLX peak {mx.get_peak_memory() / GiB:.1f} GiB", flush=True)

"""8-layer speed test for GLM 5.3 Flash on mlx-vlm 0.7.12.

What it measures: decode speed of the real generation loop (mlx-vlm's stream_generate, temperature 0.7, top_p 0.95)
on the first K decoder layers only, with patches.py applied. The text it writes is garbage (8 of 45 layers), by
design: the point is milliseconds per token including the output layer, the sampler and the loop overhead, in
seconds instead of the hours a full-model run takes. Faster per-layer here means faster on the full model.

Measured on a quiet, cool M3 Ultra (K=8, 300 tokens): about 113 tok/s with the four fixes; with the attention and
output layer stored at 8-bit (SPEED=8) about 145; at 4-bit attention (SPEED=4) about 159. With bugs 1 to 3
fixed but not bug 4, one token took 34.7 ms on this slice; with bug 4 fixed too, 9.0 ms.
Run it a few times back to back: if the speed slides after a few minutes and recovers after a rest, the GPU is
being slowed by heat (check `sudo powermetrics --samplers gpu_power,thermal -i 2000 -n 5`), not the code.

Memory: about 32 GB peak for K=8 (lazy load: only the kept layers' weights are read), plus about 5 GB with MTP=1.
A guard thread exits if system free memory drops under 30% or MLX passes STOP_GB (default 50).

Run (from this folder):
  ~/venvs/mlx-vlm-glm5/bin/python -I slice_speed.py [K=8] [TOKENS=300]
Switches (environment variables):
  SPEED=8 | SPEED=4  speed.shrink_attention(model, bits): linear attention at that many bits, output layer at 8-bit.
  MTP=1              also load the model's own draft head (needs bug 5's fix) and decode speculatively.
  BLOCK=n            draft block size for MTP (mlx-vlm's default otherwise).
  TOP_K=6            route each token to 6 experts instead of the trained 8 (about 4% faster on the slice, but it
                     changes the top next word 3.7% of the time, so it's an experiment, not a setting).
  TEMP, TOP_P        sampling settings (default 0.7, 0.95).
  NOPATCH=1          bug 1's load fix only, to see the slow speed. Short prompt only: without bug 2's fix a long
                     prompt can take far more memory.
The model path is the one in patches.py (~/models/GLM-5.3-Flash-mlx-4bit).
"""
import os, re, subprocess, sys, threading, time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.expanduser("~/models/GLM-5.3-Flash-mlx-4bit")  # same path as patches.py
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
STOP_GB = float(os.environ.get("STOP_GB", "50"))
NOPATCH = os.environ.get("NOPATCH") == "1"
if NOPATCH and (os.environ.get("SPEED") or os.environ.get("MTP") == "1"):
    sys.exit("SPEED and MTP need the fixes (MTP needs bug 5's): don't combine them with NOPATCH=1")
if os.environ.get("BLOCK") and os.environ.get("MTP") != "1":
    print("note: BLOCK only matters with MTP=1; ignored", flush=True)
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
if NOPATCH:  # bug 1 only: without it the model won't load
    import mlx_vlm.models.glm5_next.glm5_next as _g5
    _g5.Model.quantization_path_aliases = lambda self, p: (
        (p.removeprefix("language_model.").replace(".gate_up_proj", ".gate_proj"),) if ".gate_up_proj" in p else ())
else:
    import patches  # noqa: F401  (must come before load())
    import speed
import mlx_vlm.models.glm5_next.language as L  # noqa: E402
from mlx_vlm import load, stream_generate  # noqa: E402


def set_top_k(model, k):
    """Route each token to its top k routed experts instead of 8 (the shared expert always runs). The gate
    renormalises the chosen weights, so the k carry the full weight. Main model only; a draft head keeps 8."""
    n = 0
    for _, m in model.language_model.named_modules():
        if isinstance(m, L.MoEGate):
            m.top_k = k
            n += 1
    return n


model, processor = load(MODEL_DIR, lazy=True)
lm = model.language_model
lm.model.layers = lm.model.layers[:K]
print(f"fixes: {'bug 1 only' if NOPATCH else 'all (patches.py)'}; first {K} layers", flush=True)
if not NOPATCH and os.environ.get("SPEED"):
    speed.shrink_attention(model, bits=int(os.environ["SPEED"]))
    print(f"speed setup: linear attention at {os.environ['SPEED']}-bit, output layer at 8-bit", flush=True)
if os.environ.get("TOP_K"):
    print(f"top_k: {os.environ['TOP_K']} experts in {set_top_k(model, int(os.environ['TOP_K']))} MoE layers", flush=True)
mx.eval(lm.model.parameters(), lm.lm_head.parameters())
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
GEN = {}
if os.environ.get("MTP") == "1":
    from mlx_vlm.speculative.drafters import load_drafter
    t1 = time.time()
    draft, kind = load_drafter(MODEL_DIR, kind="mtp")
    mx.eval(draft.parameters())
    print(f"draft head {type(draft).__name__} kind={kind} loaded in {time.time() - t1:.0f}s, "
          f"active {mx.get_active_memory() / GiB:.1f} GiB", flush=True)
    GEN = {"draft_model": draft, "draft_kind": kind}
    if os.environ.get("BLOCK"):
        GEN["draft_block_size"] = int(os.environ["BLOCK"])
text = tok.apply_chat_template([{"role": "user", "content": "Explain photosynthesis."}], tokenize=False,
                               add_generation_prompt=True)
last = None
for r in stream_generate(model, processor, text, max_tokens=N, temperature=float(os.environ.get("TEMP", "0.7")),
                         top_p=float(os.environ.get("TOP_P", "0.95")), **GEN):
    last = r
print(f"K={K}: {last.generation_tokens} tok at {last.generation_tps:.1f} tok/s = {1000 / last.generation_tps:.1f} ms/token "
      f"({1000 / last.generation_tps / K:.2f} ms/layer incl. the tail); prompt {last.prompt_tokens} tok at "
      f"{last.prompt_tps:.0f} tok/s; peak {last.peak_memory:.1f} GB", flush=True)

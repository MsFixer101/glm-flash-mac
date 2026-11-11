"""Watch the residual stream's number type (bug 3), for learning.

What it shows: this model passes four running totals from layer to layer (the "hyper-connection" streams, 4 x 4096
numbers per token), and mlx-vlm keeps them in float32 so the mixing between them stays stable. Without the fixes
those float32 numbers go straight into every attention, MLP and expert block, and every bf16 weight they meet is
upgraded to float32 too: four times the work per multiply. patches.py feeds each block bf16 and hands its output
back as float32, so only the mixing stays float32.

The script loads the first 4 layers lazily, sends one short prompt through them, and prints the number type
  (a) handed from layer to layer (the residual stream),
  (b) arriving at the expert block inside each MoE layer (layer 3 is the first one),
  (c) coming out of the stack.
Expected: (a) float32 both ways; (b) float32 without the fixes, bfloat16 with them; (c) float32 without the fixes,
bfloat16 with them (bug 4b's fix: the output layer is bf16).

Memory: about 10 GiB (MLX peak). A guard thread exits if system free memory drops under 30% or MLX passes STOP_GB (default 30).

Run (from this folder):
  ~/venvs/mlx-vlm-glm5/bin/python -I dtype_watch.py              (with the fixes)
  NOPATCH=1 ~/venvs/mlx-vlm-glm5/bin/python -I dtype_watch.py    (without: bug 1's load fix only)
The model path is the one in patches.py (~/models/GLM-5.3-Flash-mlx-4bit).
"""
import os, re, subprocess, sys, threading, time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.expanduser("~/models/GLM-5.3-Flash-mlx-4bit")  # same path as patches.py
STOP_GB = float(os.environ.get("STOP_GB", "30"))
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
if os.environ.get("NOPATCH") == "1":  # only bug 1's load fix (it won't load without it), none of the others
    import mlx_vlm.models.glm5_next.glm5_next as _g5
    _g5.Model.quantization_path_aliases = lambda self, p: (
        (p.removeprefix("language_model.").replace(".gate_up_proj", ".gate_proj"),) if ".gate_up_proj" in p else ())
else:
    import patches  # noqa: F401  (must come before load())
from mlx_vlm import load  # noqa: E402
from mlx_vlm.models.switch_layers import SwitchGLU  # noqa: E402
import mlx_vlm.models.glm5_next.language as L  # noqa: E402

between = []
_dl = L.DecoderLayer.__call__


def _dl_watch(self, x, *a, **k):
    r = _dl(self, x, *a, **k)
    between.append(r[0].dtype if isinstance(r, tuple) else r.dtype)
    return r


L.DecoderLayer.__call__ = _dl_watch

seen = []
_orig = SwitchGLU.__call__


def _watch(self, x, *a, **k):
    seen.append(x.dtype)
    return _orig(self, x, *a, **k)


SwitchGLU.__call__ = _watch

print(f"fixes: {'bug 1 only' if os.environ.get('NOPATCH') == '1' else 'all (patches.py)'}", flush=True)
model, processor = load(MODEL_DIR, lazy=True)
lm = model.language_model
lm.model.layers = lm.model.layers[:4]
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
ids = mx.array([tok.encode("Why is the sky blue?")])
out = lm.model(ids)
mx.eval(out)
print(f"residual stream handed on by each layer: {[str(d) for d in between]}")
print(f"numbers arriving at the expert block, per MoE layer: {[str(d) for d in seen]}")
print(f"final hidden state out of the stack: {out.dtype}")
print(f"MLX peak memory: {mx.get_peak_memory() / GiB:.1f} GiB")

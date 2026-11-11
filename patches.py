"""Fixes for GLM 5.3 Flash (mlx-community/GLM-5.3-Flash-4bit) on mlx-vlm 0.7.12. Import before load()."""
import json, os
import mlx.core as mx
import mlx.nn as nn

MODEL_DIR = os.path.expanduser("~/models/GLM-5.3-Flash-mlx-4bit")

# Bug 1 (won't load): mlx-vlm fuses each shared expert's gate_proj + up_proj into gate_up_proj, but the
# checkpoint lists their bit widths under the separate names. Point the fused name back at gate_proj,
# after checking gate and up always share the same settings.
import mlx_vlm.models.glm5_next.glm5_next as _g5
_q = json.load(open(os.path.join(MODEL_DIR, "config.json")))["quantization"]
for _k, _v in _q.items():
    if isinstance(_v, dict) and _k.endswith(".gate_proj"):
        assert _q.get(_k[:-len("gate_proj")] + "up_proj") == _v, f"gate/up quantization differ at {_k}"
_g5.Model.quantization_path_aliases = lambda self, p: (
    (p.removeprefix("language_model.").replace(".gate_up_proj", ".gate_proj"),) if ".gate_up_proj" in p else ())

# Bug 2 (memory spike): sparse attention reached its fused kernel in float32, which it refuses, and fell
# back to a huge copy. Hand the kernel bf16, give the result back in the original type.
import mlx_vlm.models.glm5_next.language as _gl
_sparse_orig = _gl._sparse_prefill_attention
def _sparse_bf16(q, k, v, indices, scale, **kw):
    bf = mx.bfloat16
    return _sparse_orig(q.astype(bf), k.astype(bf), v.astype(bf), indices, scale, **kw).astype(q.dtype)
_gl._sparse_prefill_attention = _sparse_bf16

# Bug 3 (float32 into every block): the hyper-connection streams are float32, so attention, MLP and
# expert blocks all ran in float32. Feed each block bf16, and hand its output back as float32 so the
# hyper-connection mixing stays float32 (half the drift of bf16 streams, same speed).
for _cls in (_gl.Glm5NextLinearAttention, _gl.Glm5NextAttention, _gl.Glm5NextMLP, _gl.Glm5NextMoE):
    def _mk(orig):
        def f(self, x, *a, **k):
            r = orig(self, x.astype(mx.bfloat16), *a, **k)
            return (r[0].astype(mx.float32), *r[1:]) if isinstance(r, tuple) else r.astype(mx.float32)
        return f
    _cls.__call__ = _mk(_cls.__call__)

# Bug 4a (fp16 scales x bf16 inputs -> float32): run each 4-bit multiply in fp16, the scales' own
# type, so the scales aren't rounded, and hand back bf16.
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear as _QSL
for _cls in (nn.QuantizedLinear, _QSL):
    def _mk16(orig):
        def f(self, x, *a, **k):
            if self["scales"].dtype == mx.float16 and x.dtype == mx.bfloat16:
                return orig(self, x.astype(mx.float16), *a, **k).astype(mx.bfloat16)
            return orig(self, x, *a, **k)
        return f
    _cls.__call__ = _mk16(_cls.__call__)

# Bug 4b (117 ms per token): the final hidden state reached the bf16 output layer in float32.
_tm_call = _gl.Glm5NextTextModel.__call__
_gl.Glm5NextTextModel.__call__ = lambda self, *a, **k: _tm_call(self, *a, **k).astype(mx.bfloat16)

# Bug 5 (draft head won't load; only matters if you use it): load_drafter() renames the guessing layer to
# "mtp_block.*" but looks its bit widths up under the old name, so its 5- and 6-bit parts are read as 4-bit.
# Copy each entry to the new name (and the fused gate_up_proj name, as in bug 1).
from mlx_vlm.speculative.drafters.glm5_next_mtp.split import Glm5NextMTPSplitter as _MTPS
_mtp_quant = _MTPS.quantization
def _mtp_quant_aliased(self, weights, source_config, text_config, quant_opts):
    q = _mtp_quant(self, weights, source_config, text_config, quant_opts)
    if not isinstance(q, dict):
        return q
    q, n = dict(q), int(text_config["num_hidden_layers"])
    for k, v in list(q.items()):
        for pre in (f"model.language_model.layers.{n}.", f"language_model.model.layers.{n}.",
                    f"model.layers.{n}.", f"layers.{n}."):
            if isinstance(v, dict) and k.startswith(pre):
                rest = k[len(pre):]
                q["mtp_block." + rest] = v
                if rest.endswith("shared_experts.gate_proj"):
                    q["mtp_block." + rest.replace("gate_proj", "gate_up_proj")] = v
    return q
_MTPS.quantization = _mtp_quant_aliased

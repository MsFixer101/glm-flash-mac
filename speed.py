"""Optional speed setup for GLM 5.3 Flash on mlx-vlm 0.7.12. Import after patches.py; call after load()."""
import mlx.core as mx
import mlx.nn as nn


def shrink_attention(model, bits=8):
    """Store the linear-attention layers and the output layer at 8-bit, in memory. Files on disk are untouched."""
    lm = model.language_model

    def pick(path, m):
        if not isinstance(m, nn.Linear) or isinstance(m, nn.QuantizedLinear) or ".self_attn." not in path:
            return False
        if path.rsplit(".", 1)[-1] not in ("qkv_proj", "o_proj"):
            return False
        layer = int(path.split("layers.")[1].split(".")[0])
        return {"group_size": 64, "bits": bits} if lm.config.layer_types[layer] == "linear_attention" else False

    nn.quantize(lm.model, class_predicate=pick)
    lm.lm_head = nn.QuantizedLinear.from_linear(lm.lm_head, group_size=64, bits=8)
    mx.eval(model.parameters())  # do it here, on the main thread: generation runs in another one

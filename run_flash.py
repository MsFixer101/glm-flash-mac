import mlx.core as mx
import patches                      # must come before load()
from mlx_vlm import load, stream_generate

mx.set_memory_limit(205 * 2**30)    # a guideline for MLX, not a hard stop
mx.set_cache_limit(2 * 2**30)

model, processor = load(patches.MODEL_DIR)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
text = tok.apply_chat_template([{"role": "user", "content": "Why is the sky blue?"}],
                               tokenize=False, add_generation_prompt=True)
for r in stream_generate(model, processor, text, max_tokens=4096, temperature=0.7, top_p=0.95):
    print(r.text, end="", flush=True)
print(f"\n\n{r.generation_tokens} tokens at {r.generation_tps:.1f} tok/s, peak {r.peak_memory:.1f} GB")

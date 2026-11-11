# GLM 5.3 Flash on a 256 GB Mac Studio (MLX, 4-bit)

Companion code for the blog post about getting GLM 5.3 Flash up to speed on a Mac Studio:
https://christinasouch.com/blog/frontier-ish-at-home-getting-glm-53-flash-up-to-speed-on-a-mac-studio

GLM 5.3 Flash (321B parameters, mixture of experts) runs on a 256 GB M3 Ultra Mac Studio in the
[mlx-community 4-bit build](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit) with
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.12, but out of the box it has five bugs: it won't load, its first
prompt can run the Mac out of memory, two separate float32 leaks make it about ten times slower than it should be,
and its draft head won't load. `patches.py` fixes all five without touching the library's files. The rest of this
repo is the launcher and the test scripts used to find and check the fixes.

The bugs are reported upstream at ISSUE_URL.

## Results

- About 2 tokens a second with only the load fix; 20.5 with the four fixes; about 26 with the speed setup (`speed.py`: 8-bit
  attention and output layer, plus the model's own draft head).
- On 200 MMLU-Pro questions it scores the same as Z.ai's own FP8 Flash in the cloud (178/200 plain on both).

## Setup

You'll need a 256 GB Mac (the model peaks around 192–194 GiB while it runs, a few GB more with the draft head),
about 204 GB of free disk, and Python 3.11 (`brew install python@3.11`). Use a fresh virtual environment and pin
these versions, because the patch wraps 0.7.12's functions by name:

```bash
python3.11 -m venv ~/venvs/mlx-vlm-glm5
~/venvs/mlx-vlm-glm5/bin/pip install \
  "git+https://github.com/Blaizzy/mlx-vlm.git@7230940302cbe602054650f18a2f77db945bc57a" \
  "mlx==0.32.4" "transformers==5.22.1"
~/venvs/mlx-vlm-glm5/bin/hf download mlx-community/GLM-5.3-Flash-4bit --local-dir ~/models/GLM-5.3-Flash-mlx-4bit
```

Every script expects the model at `~/models/GLM-5.3-Flash-mlx-4bit`. If yours is elsewhere, change `MODEL_DIR` at
the top of `patches.py` and in the script you're running.

Then, from this folder:

```bash
./launch.sh ~/venvs/mlx-vlm-glm5/bin/python run_flash.py
```

Expect a long think before the answer: the model's chat template turns thinking on at maximum effort by default.

## Safety: read this first

- **Import `patches.py` before you load the model, every time.** Without bug 2's fix, the very first prompt can run a
  256 GB Mac out of memory: on a 300-token prompt MLX went from 186 to 226 GiB in two seconds. Don't run the full
  model without it "to see the difference".
- **MLX's memory limit is a guideline, not a hard stop.** `mx.set_memory_limit` didn't stop that spike. Run the full
  model through `launch.sh`, which watches system free memory every second from outside the job and stops it if free
  memory drops below 10%.
- **Test on a slice.** Every test script here loads the model lazily and keeps only its first few layers, so it needs
  10–32 GB instead of 190, and each has its own guard that exits under 30% free memory. Run them one at a time, and
  never next to the full model.
- The fixes are written for exactly mlx-vlm 0.7.12 at the commit above. If you upgrade, check the speed and the
  answers again before trusting them.

## Files

| File | What it does | Memory |
|---|---|---|
| `patches.py` | The five fixes. Import it before `load()`. Exactly as printed in the post. | none of its own |
| `speed.py` | Optional speed setup: linear-attention layers and output layer stored at 8-bit in memory (files on disk untouched). Call `shrink_attention(model)` after `load()`. | saves ~4.5 GiB on the full model |
| `run_flash.py` | Loads the full model with the fixes and answers one question. Exactly as printed in the post. | full model: ~192–194 GiB peak |
| `launch.sh` | Runs any command under a memory watchdog: logs free % and the job's memory every second, stops the job (TERM, then KILL after 5 s) below `GUARD_FREE_PCT` (default 10), prints the lowest free % and the job's exit code. | none |
| `memory_test.py` | Layer-by-layer memory test: MLX memory after each layer's weights and forward pass, extra memory per attention/expert call, and whether the sparse-attention kernel accepted its inputs. `FIXES=0` reproduces bug 2 on the slice. | K=4 ~11 GiB; K=8 ~30 GiB; `FIXES=0` K=4 ~31 GiB |
| `dtype_watch.py` | Prints the number type of the residual stream and of what reaches the expert block, with the fixes and without (`NOPATCH=1`): bfloat16 at the experts with them, float32 without. | ~10 GiB |
| `speed_diag.py` | Number-format audit of one decode token (every block's input/output type, every 4-bit multiply's input/scales pair), decode ms/token at short and long context, ms per part, and the output layer + sampler tail. Switches compare fix sets and scale handling. | K=4 ~12 GiB; K=8 ~30 GiB |
| `slice_speed.py` | 8-layer speed test through the real generation loop. ~113 tok/s with the fixes, ~145 with `SPEED=8`. `MTP=1` adds the draft head. | K=8 ~32 GB (+5 GB with `MTP=1`) |
| `requant_check.py` | Drift check for quantising the attention and output layer: hidden-state cosine, same top next token, KL. 8-bit: cos 0.99989, top-1 100%. | K=8 ~29 GiB |

Each test script's docstring says what it measures, how to run it and its switches. Run the test scripts as
`~/venvs/mlx-vlm-glm5/bin/python -I <script>.py` from this folder; run `run_flash.py` without `-I` (it relies on
Python finding `patches.py` next to it).

## Versions

mlx 0.32.4, mlx-vlm 0.7.12 (git @7230940), transformers 5.22.1, Python 3.11. Model: mlx-community/GLM-5.3-Flash-4bit
(204 GB, mixed 4/5/6-bit, groups of 64). Measured on a Mac Studio M3 Ultra, 256 GB.

## Licence

MIT (see `LICENSE`).

# llm-benchmarking

Throughput, latency and quality benchmarks for an LLM served over an OpenAI-compatible HTTP API.

- `bench.py` measures performance: TTFT, prefill and decode speed, concurrency, prefix caching and speculative-decoding (MTP) acceptance. Python standard library only.
- `quality.sh` is a quality spot-check: GSM8K (chain-of-thought, 0-shot) and IFEval via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Both are plain HTTP clients, so they run on any OS and don't need a GPU on the machine running them. The server can be anywhere you can reach.

`results.csv` holds reference results for GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark with the TensorFold server.

## Server requirements

The scripts were written for **TensorFold** and use a few of its extensions on top of the OpenAI API:

| Uses | For | Without it |
|---|---|---|
| `POST /v1/chat/completions` (streaming, `stream_options.include_usage`) | everything | required |
| `POST /tokenize` returning `{"count": N}` | sizing filler prompts to an exact token length | `bench.py` fails at startup (vLLM's endpoint is compatible) |
| `GET /health` returning JSON with a `busy` field | waiting for the server to go idle between runs | `bench.py` fails if the body isn't JSON (vLLM returns an empty body). JSON without `busy` is fine |
| `tensorfold` object in the final stream chunk (`prefill_s`, `decode_s`, `drafted`, `accepted`) | server-side prefill/decode speed and MTP acceptance | those columns are empty. TTFT, end-to-end time and aggregate tokens/s still work |
| `ignore_eos`, `chat_template_kwargs.enable_thinking` request fields | fixed output lengths, thinking on/off | ignored or rejected depending on the server |

`quality.sh` only needs the standard chat completions endpoint, so it works against any OpenAI-compatible server.

## Setup

You need Python 3.9 or newer and a running server.

```sh
git clone https://github.com/gsilvr/llm-benchmarking.git
cd llm-benchmarking
```

**Point it at your server and model.** The defaults are `http://localhost:8888` and `GLM-5.3-Flash-EXL3`.

- URL: pass `--url http://host:port` to `bench.py`, or set `URL=http://host:port` for `quality.sh`.
- Model: there's no flag yet. Edit `MODEL` at the top of `bench.py` and `model=` in `quality.sh` to match what your server reports at `/v1/models`.

Check that `bench.py`'s parsing works (no server needed):

```sh
python3 bench.py --selftest
```

## Performance: `bench.py`

```sh
python3 bench.py --preset quick      # ~5 min: concurrency {1,4} × prompt {1k,8k}
python3 bench.py --preset decode     # content type (prose/code/json/math/translate) × concurrency {1,2,4,8}, plus thinking on
python3 bench.py --preset prefill    # TTFT/prefill curve from 128 to 128k tokens, plus concurrent 32k prefills
python3 bench.py --preset cache      # cold vs warm prefix at 8k/32k/128k
python3 bench.py --preset all        # all of the above, ~1–2 h
python3 bench.py --preset long       # opt-in: 256k / 512k / 1M prompts (needs a model with that much context)
```

Options: `--url`, `--reps N` (default 3), `--csv FILE` (default `results.csv`).

Each request is appended as one row to the CSV, and a median/p90 summary per scenario is printed at the end. Use `--csv mine.csv` to keep your runs separate from the reference results.

Key columns:

| Column | Meaning |
|---|---|
| `ttft_s`, `e2e_s` | time to first token and total request time, measured by the client |
| `prefill_tps`, `decode_tps` | prompt and generation tokens/s from the server's own timing |
| `agg_tps` | total output tokens/s across all concurrent requests in that run |
| `accept_rate` | speculative-decoding acceptance (`accepted / drafted`) |
| `cached_tok` | prompt tokens served from the prefix cache |
| `error` | the exception if a request failed. The sweep keeps going |

Notes:
- `quick`, `prefill` and `cache` set `ignore_eos` so every request generates a fixed number of tokens.
- Each prompt gets a unique random prefix, so nothing hits the prefix cache by accident, except the warm half of `cache`.
- In warm `cache` rows, `prefill_tps` counts cached tokens too, so it's inflated. Compare `ttft_s` and `cached_tok` instead.
- Concurrency 8 is above the reference server's 4 slots, so it measures queueing. Adjust the levels in `cells()` for your server.

## Quality: `quality.sh`

```sh
python3 -m venv .venv
.venv/bin/pip install "lm-eval[api]" langdetect immutabledict nltk
./quality.sh          # 200 samples per task
./quality.sh 50       # quicker
./quality.sh all      # full datasets
```

Results and per-sample logs go in `quality/`. For GSM8K, read the `strict-match` row. The script tells the model to end with "The answer is N." because the looser `flexible-extract` filter just takes the last number in the reply and misgrades answers.

It runs 4 requests at a time with up to 8192 generated tokens each. Change `num_concurrent` and `max_gen_toks` in the script to suit your server.

## Platform notes

- **Linux:** works as is.
- **macOS:** `bench.py` works as is. `./quality.sh all` fails under the built-in bash 3.2 ("unbound variable"). Run `brew install bash`, or use a numeric limit.
- **Windows:** `bench.py` works with `python bench.py ...`. Run `quality.sh` under WSL. Under Git Bash, change `.venv/bin/lm_eval` in the script to `.venv/Scripts/lm_eval`.

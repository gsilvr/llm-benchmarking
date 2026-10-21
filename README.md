# GLM-5.3-Flash-EXL3 benchmarks (2× DGX Spark, TensorFold @ localhost:8888)

## Performance — `bench.py` (stdlib only)
```
python3 bench.py --selftest
python3 bench.py --preset quick      # ~5 min: conc {1,4} × prompt {1k,8k}
python3 bench.py --preset decode     # content type (prose/code/json/math/translate) × conc {1,2,4,8}, + thinking on
python3 bench.py --preset prefill    # TTFT/prefill curve 128 → 128k tokens, + concurrent 32k prefills
python3 bench.py --preset cache      # cold vs warm prefix at 8k/32k/128k
python3 bench.py --preset all        # all of the above, ~1–2 h
python3 bench.py --preset long       # opt-in: 256k / 512k / 1M prompts (~16 min for 1M)
```
Rows are appended to `results.csv` (one per request), and a median/p90 summary prints at the end. Timing comes from the server's own `tensorfold` stats (`prefill_s`, `decode_s`, MTP `accepted/drafted`). TTFT/e2e are measured by the client.

Notes:
- `quick`/`prefill`/`cache` use `ignore_eos` to get fixed output lengths.
- In warm cache rows, `prefill_tps` counts the cached tokens too, so it's inflated. Use `ttft_s` and `cached_tok` instead.
- Concurrency 8 is above the server's 4 slots, so it measures queueing.

## Quality — lm-eval
```
python3 -m venv .venv && .venv/bin/pip install "lm-eval[api]" langdetect immutabledict nltk
./quality.sh        # GSM8K (CoT, 0-shot) + IFEval, 200 samples each; ./quality.sh all for full sets
```
Results go in `quality/`. Read the `strict-match` row for GSM8K.

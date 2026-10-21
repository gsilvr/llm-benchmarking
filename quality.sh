#!/usr/bin/env bash
# Quality spot-check via lm-eval. Usage: ./quality.sh [limit]   (default 200; "all" for full sets)
set -euo pipefail
cd "$(dirname "$0")"
LIMIT=${1:-200}; [[ $LIMIT == all ]] && LIM=() || LIM=(--limit "$LIMIT")
ARGS=(--model local-chat-completions --apply_chat_template --gen_kwargs max_gen_toks=8192 --output_path quality/ --log_samples
      --model_args model=GLM-5.3-Flash-EXL3,base_url=${URL:-http://localhost:8888}/v1/chat/completions,num_concurrent=4,max_retries=3,tokenized_requests=False,timeout=1800)
# flexible-extract grabs the last number in the reply ("7 dozens in 4 weeks" -> 4), so pin the answer format
.venv/bin/lm_eval "${ARGS[@]}" "${LIM[@]}" --tasks gsm8k_cot_zeroshot \
  --system_instruction "End your reply with a final line in exactly this plain-text form, no markdown: The answer is N."
.venv/bin/lm_eval "${ARGS[@]}" "${LIM[@]}" --tasks ifeval

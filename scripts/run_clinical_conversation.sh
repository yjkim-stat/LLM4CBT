#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Error: OPENAI_API_KEY environment variable is not set." >&2
  exit 1
fi

turn_limit=5
memory_turns=2
scenario_id="pancreatic_cancer_advanced"
python "run_clinical_conversation.py"\
  --openai_api_key $OPENAI_API_KEY\
  --config "GPTStory/${scenario_id}.yml"\
  --output_dir "outputs/clinical"\
  --scenario_id $scenario_id \
  --turn_limit $turn_limit \
  --memory_turns $memory_turns

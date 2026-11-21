#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash run_baseline.sh VALIDATION_FILE TEST_FILE [OUTPUT_DIR] [EVAL_OPTIONS...]

Runs both Baseline evaluations. Inputs can be CSV or JSONL.
Default output directory: results/ (validation/ and test/ must be new).

  bash run_baseline.sh validation.jsonl test.jsonl
  bash run_baseline.sh validation.csv test.csv results_smoke --limit 2
  bash run_baseline.sh validation.csv test.csv results_demo --demo

Set PYTHON_BIN to select a Python interpreter (default: python3).
Install dependencies with: python3 -m pip install -r requirements.txt
USAGE
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

if (( $# < 2 )); then
    usage >&2
    exit 2
fi

validation_file=$1
test_file=$2
shift 2
output_dir=results
if (( $# )) && [[ $1 != --* ]]; then
    output_dir=$1
    shift
fi

for dataset in "$validation_file" "$test_file"; do
    if [[ ! -f "$dataset" || ! -r "$dataset" ]]; then
        printf 'Cannot read dataset: %s\n' "$dataset" >&2
        exit 2
    fi
done

for split in validation test; do
    if [[ -e "$output_dir/$split" || -L "$output_dir/$split" ]]; then
        printf 'Output already exists: %s\n' "$output_dir/$split" >&2
        exit 2
    fi
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}
export MPLBACKEND=${MPLBACKEND:-Agg}

printf '\n[Validation] %s\n' "$validation_file"
"$python_bin" -u "$script_dir/baseline_eval.py" "$validation_file" "$output_dir/validation" "$@" --split validation

printf '\n[Test] %s\n' "$test_file"
"$python_bin" -u "$script_dir/baseline_eval.py" "$test_file" "$output_dir/test" "$@" --split test

printf '\nSummaries: %s/validation/overall.csv and %s/test/overall.csv\n' "$output_dir" "$output_dir"

#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SIMWORD_PYTHON:-${ALIGNMENT_EXPERIMENT_PYTHON:-python3.11}}" "$root/experiment.py" "$@"

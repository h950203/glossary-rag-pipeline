#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Preserve the old wrapper's PP-relative input paths, e.g. data/train/의학/간호학.en.
cd "$script_dir/.."
if [[ -n "${WORD_ALIGNMENT_PYTHON:-${ALIGNMENT_PYTHON:-}}" ]]; then
  exec "${WORD_ALIGNMENT_PYTHON:-$ALIGNMENT_PYTHON}" "$script_dir/align.py" "$@"
fi
if [[ -x "${WORD_ALIGNMENT_VENV:-$script_dir/.venv-bge}/bin/python" ]]; then
  exec "${WORD_ALIGNMENT_VENV:-$script_dir/.venv-bge}/bin/python" "$script_dir/align.py" "$@"
fi
if [[ -x "${ALIGNMENT_BGE_VENV:-$script_dir/../.venv-bge}/bin/python" ]]; then
  exec "${ALIGNMENT_BGE_VENV:-$script_dir/../.venv-bge}/bin/python" "$script_dir/align.py" "$@"
fi
exec conda run --no-capture-output -n "${ALIGNMENT_CONDA_ENV:-nmt}" python "$script_dir/align.py" "$@"

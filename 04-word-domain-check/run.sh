#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  set -- --input-file "$1" "${@:2}"
fi
if [[ -n "${DOMAIN_CHECK_PYTHON:-}" ]]; then
  exec "$DOMAIN_CHECK_PYTHON" "$script_dir/review.py" "$@"
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == "${DOMAIN_CHECK_CONDA_ENV:-nmt}" ]]; then
  exec python "$script_dir/review.py" "$@"
fi
exec conda run --no-capture-output -n "${DOMAIN_CHECK_CONDA_ENV:-nmt}" python "$script_dir/review.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${ALIGNMENT_BGE_VENV:-$project_dir/.venv-bge}"
conda run --no-capture-output -n "${ALIGNMENT_CONDA_ENV:-nmt}" \
  python -m venv --system-site-packages "$venv_dir"
"$venv_dir/bin/python" -m pip install \
  --index-url "${BGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}" \
  'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0'
"$venv_dir/bin/python" -m pip install -r "$project_dir/requirements.txt"
# CUDA wheels find binaries relative to this venv. Install required NVIDIA
# wheels locally when pip skipped a same-version package inherited from nmt.
"$venv_dir/bin/python" - <<'PY'
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
from packaging.requirements import Requirement
import subprocess
import sys
missing = []
for item in distribution("torch").requires or []:
    req = Requirement(item)
    if not req.name.startswith("nvidia-") or (req.marker and not req.marker.evaluate()):
        continue
    try:
        local = Path(distribution(req.name).locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve())
    except PackageNotFoundError:
        local = False
    if not local:
        missing.append(req.name + str(req.specifier))
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--ignore-installed", "--no-deps", *missing])
PY
"$venv_dir/bin/python" -c 'import torch; print("BGE environment:", torch.__version__, "CUDA:", torch.cuda.is_available())'

"$venv_dir/bin/python" "$project_dir/setup_resources.py"

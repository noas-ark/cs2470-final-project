#!/usr/bin/env bash
# Build elephant-1's runtime venv (runtimes/elephant/.venv) on a compute node, never the login node:
#   sbatch runtimes/elephant/setup.sh          (after: sbatch scripts/download_model.sh baselabs/elephant-1 <rev>)
# 1. OneJev at the commit elephant-1's setup.sh pins, into runtimes/elephant/OneJev (gitignored)
# 2. elephant-1's serve.patch from the downloaded weights, refused unless it is the reviewed file, then applied
# 3. uv sync of runtimes/elephant/pyproject.toml (torch 2.8.0+cu128, transformers 5.19.0, fla 0.5.2)
# 4. causal-conv1d==1.7.0 like upstream, best effort: without it transformers runs the short convolutions in
#    plain torch, which the worker reports (env_causal_conv1d = absent).
#SBATCH -J s1-elephant-setup
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/elephant-setup-%j.out
set -euo pipefail
ONEJEV=81ce62f1597c91e46767d4d02d6ac2e18534fe94
PATCH_SHA256=f234c125f1d38d82104b9e1f24e75bde51e4401928acfba2fbf5c59ef2fb46b8  # reviewed 2026-11-04
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f "$here/pyproject.toml" ]] || here="${SLURM_SUBMIT_DIR:?}/runtimes/elephant"  # sbatch copies the script
models=${MODELS_DIR:-/shared/home/$USER/models}
patch="$models/elephant-1/serve.patch"

echo "$PATCH_SHA256  $patch" | sha256sum -c --quiet || { echo "serve.patch is not the reviewed file" >&2; exit 1; }
[[ -d "$here/OneJev/.git" ]] || git clone --quiet https://github.com/OmniJev/OneJev.git "$here/OneJev"
git -C "$here/OneJev" fetch --quiet origin "$ONEJEV"
git -C "$here/OneJev" -c advice.detachedHead=false checkout --quiet -f "$ONEJEV"
git -C "$here/OneJev" clean -fdq
git -C "$here/OneJev" apply "$patch"
echo "OneJev@$(git -C "$here/OneJev" rev-parse HEAD) + serve.patch"

cd "$here"
uv sync --python 3.11
# its setup.py fetches a prebuilt wheel matching torch/CUDA/Python when one exists, else compiles (needs nvcc)
uv pip install --python .venv/bin/python setuptools wheel ninja packaging
uv pip install --python .venv/bin/python --no-build-isolation causal-conv1d==1.7.0 \
  || echo "causal-conv1d did not install; short convolutions will run in plain torch"
.venv/bin/python -c "import torch, transformers, qev.readout; print('torch', torch.__version__, 'cuda', \
torch.version.cuda, 'transformers', transformers.__version__)"
echo "elephant runtime ready: $here/.venv"

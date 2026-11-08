#!/usr/bin/env bash
# Download a Hub model on a compute node (login nodes cap virtual memory at 2 GB, which kills hf).
#   sbatch scripts/download_model.sh openjev/openjev                        -> $MODELS_DIR/openjev
#   sbatch scripts/download_model.sh Qwen/Qwen3.5-4B-Base <sha>            -> $MODELS_DIR/Qwen3.5-4B-Base
#   sbatch scripts/download_model.sh jaredpalmer/kev-4b <sha> kev-4b-qwen3 -> $MODELS_DIR/kev-4b-qwen3
# The optional second argument pins a revision (use the catalog's); the third names the directory.
#SBATCH -J s1-download
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/download-%j.out
set -euo pipefail
repo=${1:?usage: download_model.sh <org/name> [revision] [dir name]}
rev=${2:-}
dest=${MODELS_DIR:-/shared/home/$USER/models}/${3:-${repo#*/}}
uvx --from 'huggingface_hub[hf_xet]' hf download "$repo" ${rev:+--revision "$rev"} --local-dir "$dest"
if [[ -f "$dest/SHA256SUMS" ]]; then
  ( cd "$dest" && sha256sum -c --quiet SHA256SUMS ) && echo "verified $dest"
fi
echo "downloaded $repo${rev:+@$rev} -> $dest"

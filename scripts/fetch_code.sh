#!/usr/bin/env bash
# Check out a model's runtime code (a GitHub repo, not weights) at a pinned commit. Small, so it is
# fine on the login node:
#   scripts/fetch_code.sh AgentBull/bongard 3e6e6030a953dfe8d30d93aca2d0ab92b87b4a5a bongard-code
# -> $MODELS_DIR/bongard-code at that commit. Read the code before running anything from it.
set -euo pipefail
repo=${1:?usage: fetch_code.sh <owner/name> <commit> [dir name]}
rev=${2:?usage: fetch_code.sh <owner/name> <commit> [dir name]}
dest=${MODELS_DIR:-/shared/home/$USER/models}/${3:-${repo#*/}}
[[ -d "$dest/.git" ]] || git clone --quiet "https://github.com/$repo.git" "$dest"
git -C "$dest" fetch --quiet origin "$rev"
git -C "$dest" -c advice.detachedHead=false checkout --quiet "$rev"
echo "$repo@$(git -C "$dest" rev-parse HEAD) -> $dest"

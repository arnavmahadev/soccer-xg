#!/usr/bin/env bash
# Deploy to Hugging Face Spaces.
#
# GitHub renders YAML frontmatter as an ugly table, so README.md on `main` has
# none. HF Spaces *requires* it (sdk/app_port), so this script prepends
# deploy/hf-header.md only for the Space push, then restores the clean README.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree not clean — commit or stash before deploying." >&2
  exit 1
fi

cat deploy/hf-header.md README.md > README.tmp && mv README.tmp README.md
git add README.md
git commit -m "Add HF Spaces config header" --quiet
git push space HEAD:main --force
git reset --hard HEAD~1 --quiet

echo "Deployed to HF Space. GitHub README (no frontmatter) is unchanged on main."

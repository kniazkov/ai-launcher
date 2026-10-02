#!/usr/bin/env bash
set -Eeuo pipefail
AI_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$AI_SOURCE/lib/bootstrap.sh"
ai_main "qwen3-coder-next" "$AI_SOURCE" "$@"

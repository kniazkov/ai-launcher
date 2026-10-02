#!/usr/bin/env bash
set -Eeuo pipefail
# Called by sudo unshare --net. Only loopback is brought up in this namespace.
(( EUID == 0 )) || { echo 'Внутренний запуск требует root.' >&2; exit 1; }
[[ $# -ge 3 ]] || exit 2
ai_user=$1
shift
ip link set lo up
if [[ -t 0 && -t 1 ]]; then
  exec runuser --pty -u "$ai_user" -- "$@"
else
  exec runuser -u "$ai_user" -- "$@"
fi

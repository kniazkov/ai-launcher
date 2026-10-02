#!/usr/bin/env bash
set -Eeuo pipefail

# OS dependencies necessarily go through the system package manager. Runtime
# binaries, model weights and profiles are installed by launcher.py in the CWD.
ai_dependencies() {
  local manager missing=0 tool
  local -a elevate=()
  for tool in python3 curl tar zstd xz git ip unshare runuser; do
    command -v "$tool" >/dev/null 2>&1 || missing=1
  done
  if (( missing == 0 )); then return; fi
  if (( EUID != 0 )); then
    command -v sudo >/dev/null || { echo 'Нужен sudo для установки системных пакетов.' >&2; exit 1; }
    elevate=(sudo)
  fi
  if command -v apt-get >/dev/null; then
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y python3 curl tar zstd xz-utils git iproute2 util-linux ca-certificates
  elif command -v dnf >/dev/null; then
    "${elevate[@]}" dnf install -y python3 curl tar zstd xz git iproute util-linux util-linux-user ca-certificates
  elif command -v yum >/dev/null; then
    "${elevate[@]}" yum install -y python3 curl tar zstd xz git iproute util-linux ca-certificates
  elif command -v zypper >/dev/null; then
    "${elevate[@]}" zypper --non-interactive install python3 curl tar zstd xz git iproute2 util-linux ca-certificates
  elif command -v pacman >/dev/null; then
    "${elevate[@]}" pacman -S --needed --noconfirm python curl tar zstd xz git iproute2 util-linux ca-certificates
  else
    echo 'Поддерживаются apt-get, dnf, yum, zypper и pacman. Установи зависимости из README вручную.' >&2
    exit 1
  fi
}

ai_main() {
  local model=$1 base=$2
  shift 2
  if [[ ${1:-} == install && " $* " != *" --help "* ]]; then
    ai_dependencies
  fi
  command -v python3 >/dev/null || { echo 'Сначала запусти install: требуется Python 3.9+.' >&2; exit 1; }
  exec python3 "$base/lib/launcher.py" --model "$model" "$@"
}

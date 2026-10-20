#!/usr/bin/env bash
# Use this as the Aistation startup command; also install an SSH login fallback.
set -euo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${VIEWER_SUPERVISOR_PYTHON:-/usr/bin/python3}"
hook="# Motion Viewer service startup"
if [[ -f "$HOME/.bashrc" ]] && ! grep -Fq "$hook" "$HOME/.bashrc"; then
  temporary="$(mktemp "$HOME/.bashrc.viewer.XXXXXX")"
  trap 'rm -f -- "$temporary"' EXIT
  {
    printf '%s\n' "$hook"
    printf '%q %q --config %q start >/dev/null 2>&1\n' \
      "$python_bin" "$runtime_root/viewer-supervisor.py" "$runtime_root/viewer-service.json"
    cat "$HOME/.bashrc"
  } > "$temporary"
  chmod --reference="$HOME/.bashrc" "$temporary"
  mv -- "$temporary" "$HOME/.bashrc"
  trap - EXIT
fi
exec "$python_bin" "$runtime_root/viewer-supervisor.py" \
  --config "$runtime_root/viewer-service.json" start

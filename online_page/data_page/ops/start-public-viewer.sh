#!/usr/bin/env bash
# Compatibility entrypoint for manual startup and older operational instructions.
set -euo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$runtime_root/viewer-bootstrap.sh"
for ((attempt=0; attempt<30; attempt++)); do
  if /usr/bin/python3 "$runtime_root/viewer-supervisor.py" \
      --config "$runtime_root/viewer-service.json" status >/dev/null 2>&1; then
    echo viewer_and_tunnel_ready
    exit 0
  fi
  sleep 2
done
echo "Website is still recovering; inspect logs/supervisor.log" >&2
exit 1

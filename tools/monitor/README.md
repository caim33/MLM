# Aistation GPU monitor code

This package contains the MotionLLM Aistation GPU monitoring and automatic allocation scripts.

## Files

- `aistation_auto_occupy_watch.mjs`: periodically checks account usage, selects a qualifying node, creates a four-GPU environment, refreshes connection information, and updates SSH config.
- `aistation_gpu_monitor.mjs`: queries GPU availability by resource group and node.
- `aistation_user_gpu_usage.mjs`: checks whether the current account already has an active GPU environment.
- `aistation_write_ssh_info.mjs`: writes current environment connection information and updates one SSH config entry.
- `aistation_api_get.mjs`: authenticated Aistation GET helper.
- `aistation_api_post.mjs`: authenticated Aistation POST helper.

## Requirements

- Node.js with built-in `fetch` and `WebSocket` support (the deployed version uses Node.js 24).
- A Chrome instance logged into Aistation and started with remote debugging on port `9223`.
- Environment variable `AISTATION_CHROME_DEBUG_PORT=9223` when a different default is not desired.

Run `node aistation_auto_occupy_watch.mjs --help` is not supported; inspect `parseArgs` near the top of the watcher for all arguments.

No credentials, connection files, SSH config, runtime state, or logs are included in this archive.

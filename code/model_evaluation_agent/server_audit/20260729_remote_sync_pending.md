# Remote controller sync status

Updated: 2026-07-29 (Asia/Shanghai)

The local remote-controller package is complete and its end-to-end metadata
self-test passes. The planned remote root is:

`/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval`

At the time of the sync attempt, the endpoint recorded in the current
`dev_env_connection.txt` refused SSH connections. The Aistation browser CDP
endpoint used by the existing connection-refresh helper was also unavailable,
so a fresh SSH endpoint could not be retrieved safely.

No remote files were created during the failed attempt. Once the connection
file is refreshed, run `scripts/sync_to_remote.ps1`; it creates the exact
remote tree, uploads the controller, and runs `scripts/selftest_workflow.py`
without loading any model or using a GPU.

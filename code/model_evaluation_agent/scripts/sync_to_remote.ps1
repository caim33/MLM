param(
    [string]$ConnectionFile = "D:\MotionLLM\dev_env_connection.txt",
    [string]$KnownHostsFile = "D:\MotionLLM\.codex_tmp\known_hosts_31976",
    [string]$AskPass = "D:\MotionLLM\.codex_tmp\ssh_askpass.cmd",
    [string]$RemoteRoot = "/wangbenyou-sulongjie/Motion-r1/caimeng/MLLM/codex_runs/unified_model_eval"
)

$ErrorActionPreference = "Stop"
$AgentRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $ConnectionFile)) {
    throw "Connection file not found: $ConnectionFile"
}
if (-not (Test-Path -LiteralPath $KnownHostsFile -PathType Leaf)) {
    throw "Pinned known-hosts file not found: $KnownHostsFile"
}
if (-not (Test-Path -LiteralPath $AskPass)) {
    throw "SSH askpass helper not found: $AskPass"
}
if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteRoot -match '(^|/)\.\.(/|$)' -or $RemoteRoot -match '//') {
    throw "RemoteRoot must be a normalized absolute POSIX path"
}

$lines = Get-Content -LiteralPath $ConnectionFile -Encoding UTF8
function Read-Field([string]$name) {
    $line = $lines | Where-Object { $_ -like "$name=*" } | Select-Object -First 1
    if (-not $line) { throw "Missing connection field: $name" }
    return $line.Substring($name.Length + 1)
}

$hostName = Read-Field "host"
$port = Read-Field "port"
$userName = Read-Field "username"
$password = Read-Field "password"
if ($hostName -notmatch '^[A-Za-z0-9.-]+$') { throw "Invalid SSH host in connection file" }
if ($port -notmatch '^[0-9]+$' -or [int]$port -lt 1 -or [int]$port -gt 65535) {
    throw "Invalid SSH port in connection file"
}
if ($userName -notmatch '^[A-Za-z_][A-Za-z0-9._-]*$') { throw "Invalid SSH username" }
if ([string]::IsNullOrEmpty($password)) { throw "SSH password is empty" }

$hostLookup = "[$hostName]:$port"
& ssh-keygen -F $hostLookup -f $KnownHostsFile *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Pinned known-hosts file has no entry for the configured endpoint"
}

$destination = "${userName}@${hostName}:$RemoteRoot/"
$sshOptions = @(
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=$KnownHostsFile",
    "-o", "HostKeyAlgorithms=ssh-ed25519",
    "-o", "KexAlgorithms=curve25519-sha256",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
    "-p", $port
)
$scpOptions = @(
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=$KnownHostsFile",
    "-o", "HostKeyAlgorithms=ssh-ed25519",
    "-o", "KexAlgorithms=curve25519-sha256",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
    "-P", $port
)

try {
    $env:CODEX_SSH_PASSWORD = $password
    $env:SSH_ASKPASS = $AskPass
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "codex:0"

    & ssh @sshOptions "$userName@$hostName" "if [ -e '$RemoteRoot' ]; then exit 73; fi; mkdir -p '$RemoteRoot/batches' '$RemoteRoot/scripts' '$RemoteRoot/templates' '$RemoteRoot/server_audit'"
    if ($LASTEXITCODE -eq 73) {
        throw "RemoteRoot already exists; choose a fresh run directory to avoid overwriting prior work"
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create fresh remote controller directory" }

    Get-ChildItem -LiteralPath $AgentRoot -File | ForEach-Object {
        & scp @scpOptions $_.FullName $destination
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload $($_.Name)" }
    }
    foreach ($directory in ("scripts", "templates", "server_audit", "smoke_assets")) {
        & scp @scpOptions -r (Join-Path $AgentRoot $directory) $destination
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload $directory" }
    }

    & ssh @sshOptions "$userName@$hostName" "cd '$RemoteRoot' && python3 scripts/selftest_workflow.py && find . -maxdepth 2 -type f -printf '%p\n' | sort"
    if ($LASTEXITCODE -ne 0) { throw "Remote workflow self-test failed" }
    Write-Output "SYNC_OK $RemoteRoot"
}
finally {
    Remove-Item Env:CODEX_SSH_PASSWORD -ErrorAction SilentlyContinue
    Remove-Item Env:SSH_ASKPASS -ErrorAction SilentlyContinue
    Remove-Item Env:SSH_ASKPASS_REQUIRE -ErrorAction SilentlyContinue
    Remove-Item Env:DISPLAY -ErrorAction SilentlyContinue
}

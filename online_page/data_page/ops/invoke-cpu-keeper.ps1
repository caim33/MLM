param(
  [Parameter(Mandatory=$true)][string]$Config,
  [switch]$Execute,
  [switch]$Refresh,
  [switch]$ExerciseRestart
)
$ErrorActionPreference = 'Stop'
$keeperConfig = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if (-not $keeperConfig.enabled) { '{"action":"DISABLED"}'; exit 0 }
$keeperLock = $null
try {
  try {
    $keeperLock = [IO.File]::Open(
      ($keeperConfig.stateFile + '.lock'), [IO.FileMode]::OpenOrCreate,
      [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
  } catch [IO.IOException] { '{"action":"ALREADY_RUNNING"}'; exit 0 }
  $instructions = [IO.File]::ReadAllText($keeperConfig.agentsFile)
  $loginOffset = $instructions.IndexOf('aistation connection')
  if ($loginOffset -lt 0) { throw 'LOGIN_ENTRY_MISSING' }
  $loginText = $instructions.Substring($loginOffset)
  $env:AISTATION_LOGIN_USERNAME = [regex]::Match($loginText,"(?m)^user='([^']+)'").Groups[1].Value
  $env:AISTATION_LOGIN_PASSWORD = [regex]::Match($loginText,"(?m)^password='([^']+)'").Groups[1].Value
  $keeperArguments = @((Join-Path $PSScriptRoot 'aistation-cpu-keeper.mjs'), '--config', $Config)
  if ($Execute) { $keeperArguments += '--execute' }
  if ($Refresh) { $keeperArguments += '--refresh' }
  if ($ExerciseRestart) { $keeperArguments += '--exercise-restart' }
  & $keeperConfig.nodeExecutable @keeperArguments
  $keeperExitCode = $LASTEXITCODE
} catch {
  '{"action":"NEEDS_ATTENTION","reason":"LOCAL_KEEPER_FAILED"}'
  $keeperExitCode = 1
} finally {
  $loginText = $null
  $instructions = $null
  Remove-Item Env:AISTATION_LOGIN_USERNAME,Env:AISTATION_LOGIN_PASSWORD -ErrorAction SilentlyContinue
  if ($keeperLock) { $keeperLock.Dispose() }
}
exit $keeperExitCode

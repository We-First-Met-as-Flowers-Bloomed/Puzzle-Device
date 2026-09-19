$ErrorActionPreference = 'Stop'

$source = Get-Content -Raw (Join-Path $PSScriptRoot '..\hardware\four_axis.c')

if ($source -match 'START_POSITION_READ_ATTEMPTS|POSITION READ RETRY|generic_move_capture_start') {
    throw 'Normal task moves must no longer depend on start-position reads.'
}
if ($source -notmatch 'generic_move_prepare') {
    throw 'Normal moves must initialize status-only completion state.'
}
if ($source -match 'Emm42_RunReadRetries\s*\(\s*serial_command_relative') {
    throw 'Movement commands must never be retried automatically.'
}

Write-Output 'position_retry_policy: PASS'

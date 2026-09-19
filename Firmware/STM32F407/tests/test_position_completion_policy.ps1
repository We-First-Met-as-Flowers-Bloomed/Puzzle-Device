$ErrorActionPreference = 'Stop'
$source = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw
$project = Get-Content (Join-Path $PSScriptRoot '..\MDK-ARM\Automatic inspection vehicle.uvprojx') -Raw

if ($source -notmatch 'serial_read_status') {
    throw 'serial-axis completion must poll the Emm42 0x3A status'
}
if ($source -notmatch 'EMM42_STATUS_IN_POSITION') {
    throw 'serial-axis completion must require the Emm42 in-position bit'
}
if ($source -notmatch '#define\s+ARRIVAL_CONFIRM_POLLS\s+3u') {
    throw 'arrival must require three consecutive valid confirmations'
}
if ($source -notmatch 'generic_move_failed') {
    throw 'communication, stall, and timeout failures must be surfaced'
}
if ($source -match 'generic_move_capture_start') {
    throw 'normal task moves must not capture a 0x36 encoder start position'
}
if ($source -match 'AxisPosition_HasReached') {
    throw 'normal task completion must not compare encoder travel'
}
if ($source -notmatch 'DRIVER ARRIVED') {
    throw 'successful status-only completion must be explicit in diagnostics'
}
if ($source -notmatch 'MOVE_TIMEOUT_MARGIN_MS' -or
    $source -notmatch 'EMM42_STATUS_STALLED') {
    throw 'status-only completion must preserve timeout and stall failure'
}

Write-Output 'position_completion_policy: PASS'

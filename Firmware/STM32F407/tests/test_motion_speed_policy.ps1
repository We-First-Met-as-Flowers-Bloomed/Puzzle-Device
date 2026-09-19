$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw
$taskSource = Get-Content "$PSScriptRoot\..\app\task.c" -Raw

foreach ($required in @(
    '#define\s+NORMAL_XY_SPEED_RPM\s+2000u',
    '#define\s+NORMAL_Z_SPEED_RPM\s+1000u',
    '#define\s+NORMAL_R_SPEED_RPM\s+1500u',
    '#define\s+CONSERVATIVE_AXIS_SPEED_RPM\s+1000u',
    '#define\s+NORMAL_AXIS_ACCELERATION\s+30u',
    '#define\s+CONSERVATIVE_AXIS_ACCELERATION\s+10u',
    '#define\s+POSITION_POLL_MS\s+25u',
    '#define\s+ARRIVAL_CONFIRM_POLLS\s+3u'
)) {
    if ($source -notmatch $required) { throw "missing staged motion parameter: $required" }
}
foreach ($required in @(
    '#define\s+TRANSFER_XY_SPEED_RPM\s+2000u',
    '#define\s+TRANSFER_Z_SPEED_RPM\s+1000u',
    '#define\s+TRANSFER_R_SPEED_RPM\s+1500u'
)) {
    if ($taskSource -notmatch $required) { throw "missing transfer speed: $required" }
}
if ($taskSource -notmatch '\(current_task\s*==\s*1u\)\s*\|\|\s*\(current_task\s*==\s*2u\)') {
    throw 'Task 1 and Task 2 must share the speed-aware motion path'
}
if ($source -notmatch '#define\s+CALIBRATION_SERIAL_SPEED_RPM\s+300u') {
    throw 'Task 3-6 serial calibration command speed must remain 300 RPM'
}
if ($source -notmatch 'serial_command_relative_at_profile') {
    throw 'startup and homing require an explicit conservative motion profile'
}

Write-Output 'motion_speed_policy: PASS'

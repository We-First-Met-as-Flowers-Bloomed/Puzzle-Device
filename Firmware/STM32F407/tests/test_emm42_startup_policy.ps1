$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw
$header = Get-Content "$PSScriptRoot\..\hardware\four_axis.h" -Raw

foreach ($mapping in @(
    'x_serial_axis\.uart\s*=\s*&huart6',
    'y_serial_axis\.uart\s*=\s*&huart3',
    'z_serial_axis\.uart\s*=\s*&huart4',
    'r_serial_axis\.uart\s*=\s*&huart5'
)) {
    if ($source -notmatch $mapping) { throw "missing serial mapping: $mapping" }
}
if ($source -notmatch 'void\s+FourAxis_Init\s*\([^)]*\)[\s\S]*FOUR_AXIS_STARTUP_WAITING' -or
    $source -notmatch 'FOUR_AXIS_STARTUP_WAITING[\s\S]*startup_xy_zero_start\s*\(\s*\)' -or
    $source -notmatch 'startup_xy_zero_process[\s\S]*driver_homing_start\s*\(\s*\)') {
    throw 'startup must run XY measured-offset zero before saved-zero Z/R homing'
}
foreach ($required in @(
    '#define\s+STARTUP_X_ZERO_PULSES\s+160u',
    '#define\s+STARTUP_Y_ZERO_PULSES\s+640u',
    'serial_command_relative_at_profile\s*\(\s*&x_serial_axis\s*,\s*STARTUP_X_ZERO_PULSES\s*,\s*1u',
    'serial_command_relative_at_profile\s*\(\s*&y_serial_axis\s*,\s*STARTUP_Y_ZERO_PULSES\s*,\s*1u',
    'AxisLimit_Reset\s*\(\s*&x_position_limit\s*\)',
    'AxisLimit_Reset\s*\(\s*&y_position_limit\s*\)'
)) {
    if ($source -notmatch $required) { throw "missing XY startup-zero policy: $required" }
}
$xyStart = [regex]::Match(
    $source,
    'static\s+void\s+startup_xy_zero_start\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if ($xyStart.Success -and $xyStart.Groups['body'].Value -match 'Emm42_BuildSetHomingZero') {
    throw 'XY startup zero must not write driver flash'
}
foreach ($required in @(
    'recorded_home_reference\[4\]',
    'recorded_home_reference_valid\[4\]',
    'startup_capture_recorded_home\s*\(',
    '\[ZERO REF\]',
    'ZERO RECORD ERR'
)) {
    if ($source -notmatch $required) { throw "missing startup home-reference capture: $required" }
}
$capture = [regex]::Match(
    $source,
    'static\s+uint8_t\s+startup_capture_recorded_home\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if ($capture.Success -and $capture.Groups['body'].Value -match 'Emm42_BuildSetHomingZero') {
    throw 'startup reference capture must not write driver flash'
}
if ($source -notmatch 'serial_command_relative[\s\S]*serial_send_position_with_recovery\s*\(' -or
    $source -notmatch 'serial_send_position_with_recovery[\s\S]*serial_send_result\s*\([^;]+0xFDu\)') {
    throw 'position commands must consume and validate the FD acknowledgement'
}
if ($header -notmatch 'FOUR_AXIS_STARTUP_READY' -or
    $header -notmatch 'FourAxis_GetStartupState' -or
    $header -notmatch 'FourAxis_GetStartupText') {
    throw 'startup state must be public for task gating and diagnostics'
}
Write-Output 'emm42_startup_policy: PASS'

$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw

if ($source -match '\bserial_query\s*\(') {
    throw 'homing status must use the existing UART receive path, not undefined serial_query'
}

foreach ($required in @(
    '#define\s+EMM42_HOMING_MODE_SINGLE_TURN_NEAREST\s+0x00u',
    '#define\s+EMM42_HOMING_MODE_SINGLE_TURN_DIRECTIONAL\s+0x01u',
    '#define\s+EMM42_HOMING_RUNNING\s+0x04u',
    '#define\s+EMM42_HOMING_FAILED\s+0x08u',
    '#define\s+HOMING_TIMEOUT_MS\s+20000u',
    '#define\s+STARTUP_HOMING_DELAY_MS\s+1000u',
    '#define\s+HOMING_COMMAND_ATTEMPTS\s+3u',
    'driver_homing_start[\s\S]*serial_start_homing_fire_and_forget\s*\(\s*&z_serial_axis\s*,\s*EMM42_HOMING_MODE_SINGLE_TURN_NEAREST\s*\)',
    'driver_homing_start[\s\S]*serial_start_homing_with_retries\s*\(\s*&r_serial_axis\s*,\s*EMM42_HOMING_MODE_SINGLE_TURN_NEAREST\s*\)',
    'r_position_pulses',
    'x_homing_complete',
    'y_homing_complete',
    'Emm42_BuildReadHomingState',
    'Emm42_ParseHomingState',
    'serial_drain_rx\s*\(\s*axis\s*\)[\s\S]*HAL_UART_Transmit\s*\(\s*axis->uart[\s\S]*serial_receive_target_frame\s*\(\s*axis\s*,\s*0x3Bu',
    'AxisLimit_Reset\s*\(\s*&z_position_limit\s*\)'
)) {
    if ($source -notmatch $required) { throw "missing Z/R homing policy: $required" }
}

$homingStart = [regex]::Match(
    $source,
    'static\s+void\s+software_homing_start\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $homingStart.Success) { throw 'software_homing_start is missing' }
$body = $homingStart.Groups['body'].Value
if ($body -match 'serial_start_homing|Emm42_BuildReturnToZero') {
    throw 'task/exit homing must not use driver 0x9A homing'
}

$driverStart = [regex]::Match(
    $source,
    'static\s+void\s+driver_homing_start\s*\(void\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $driverStart.Success) { throw 'driver_homing_start is missing' }
if ($driverStart.Groups['body'].Value -match '&[xy]_serial_axis') {
    throw 'power-up driver homing must leave X and Y stationary'
}

$homingProcess = [regex]::Match(
    $source,
    'FourAxisHomingState_t\s+FourAxis_HomingProcess\s*\(void\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $homingProcess.Success) { throw 'FourAxis_HomingProcess is missing' }
if ($source -match 'serial_read_homing_state\s*\(\s*&z_serial_axis') {
    throw 'power-up Z must not query 0x3B completion state'
}
if ($driverStart.Groups['body'].Value -match 'z_ok|serial_start_homing_with_retries\s*\(\s*&z_serial_axis') {
    throw 'power-up Z must not wait for or validate its 0x9A reply'
}
if ($driverStart.Groups['body'].Value -notmatch 'z_homing_complete\s*=\s*1u') {
    throw 'power-up Z must be released immediately after fire-and-forget transmit'
}
foreach ($axis in 'x','y','z','r') {
    if ($body -notmatch "serial_command_relative_at_profile\s*\(\s*&$axis`_serial_axis") {
        throw "task/exit homing must return $axis by tracked relative pulses"
    }
}

$process = [regex]::Match(
    $source,
    'void\s+FourAxis_Process\s*\(void\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $process.Success -or
    $process.Groups['body'].Value -notmatch 'FOUR_AXIS_STARTUP_WAITING[\s\S]*STARTUP_HOMING_DELAY_MS[\s\S]*startup_xy_zero_start' -or
    $source -notmatch 'startup_xy_zero_process[\s\S]*driver_homing_start') {
    throw 'startup must wait, establish XY zero, and then use driver homing for Z/R'
}

Write-Output 'zr_homing_policy: PASS'

$ErrorActionPreference = 'Stop'
$axis = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw
$main = Get-Content (Join-Path $PSScriptRoot '..\Core\Src\main.c') -Raw

if ($axis -notmatch 'static\s+SerialAxis_t\s+r_serial_axis') {
    throw 'R must have a serial-axis state'
}
if ($axis -notmatch 'FOUR_AXIS_R\)\s+return\s+&r_serial_axis') {
    throw 'R axis id must resolve to UART5 serial state'
}
if ($axis -notmatch '#define\s+NORMAL_R_SPEED_RPM\s+1500u') {
    throw 'R stepper must use the approved 1500 RPM normal motion speed'
}
if ($axis -notmatch '#define\s+R_SERIAL_DIRECTION_INVERT\s+1u') {
    throw 'R default positive rotation must be inverted to clockwise'
}
if ($axis -match 'Servo_(SetAngle|Off|HardwareInit)') {
    throw 'four-axis motion driver must not call servo PWM functions'
}
if ($main -match 'Servo_HardwareInit\s*\(') {
    throw 'application must not initialize the removed R-axis servo'
}
if ($task -match 'Task_SplitRotation\s*\(') {
    throw 'R stepper must execute each requested angle as one continuous move'
}
if ($task -match 'TASK11_R_TEST_DEG') {
    throw 'Removed Task 11 must not remain selectable'
}
$transfer = [regex]::Match(
    $task,
    'static\s+void\s+load_transfer_script\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $transfer.Success) { throw 'load_transfer_script is missing' }
$rotations = [regex]::Matches($transfer.Groups['body'].Value, 'queue_rotation\s*\(')
if ($rotations.Count -ne 1) {
    throw 'piece script must rotate R only at placement; R returns in parallel with X/Y software homing'
}
if ($transfer.Groups['body'].Value -match 'queue_rotation\s*\([^,]+,\s*0u') {
    throw 'piece script must not return R before X/Y software homing starts'
}

Write-Output 'r_axis_stepper_policy: PASS'

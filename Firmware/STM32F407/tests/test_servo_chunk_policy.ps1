$ErrorActionPreference = 'Stop'
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw
$axis = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw

if ($task -match 'TASK_SERVO_MAX_DEG|Task_SplitRotation') {
    throw 'stepper R axis must not retain the servo chunk limit'
}
if ($axis -notmatch 'degrees\s*\*\s*AXIS_FULL_TURN_PULSES') {
    throw 'R angle must be converted with the 3200-pulse revolution scale'
}

Write-Output 'r_rotation_policy: PASS'

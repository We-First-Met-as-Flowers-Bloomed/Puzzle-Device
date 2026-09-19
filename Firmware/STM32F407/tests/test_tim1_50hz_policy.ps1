$ErrorActionPreference = 'Stop'
$tim = Get-Content (Join-Path $PSScriptRoot '..\Core\Src\tim.c') -Raw
$servo = Get-Content (Join-Path $PSScriptRoot '..\hardware\servo_hal.c') -Raw
$magnet = Get-Content (Join-Path $PSScriptRoot '..\hardware\electromagnet_hal.c') -Raw

if ($tim -notmatch 'htim1\.Init\.Prescaler\s*=\s*168-1') {
    throw 'TIM1 must retain a 1 MHz counter clock'
}
if ($tim -notmatch 'htim1\.Init\.Period\s*=\s*20000-1') {
    throw 'TIM1 period must be 20000 counts for 50 Hz'
}
if ($servo -notmatch 'Servo_ClampPulseUs\(pulse_us\)') {
    throw 'servo CCR must remain expressed in microseconds'
}
if ($magnet -notmatch 'Electromagnet_DutyToCompare') {
    throw 'electromagnet PWM must scale duty from the live TIM1 period'
}

Write-Output 'tim1_50hz_policy: PASS'

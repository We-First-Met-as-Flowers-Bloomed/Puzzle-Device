$ErrorActionPreference = 'Stop'
$parser = Get-Content (Join-Path $PSScriptRoot '..\app\vision_jsonl.c') -Raw
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw

if ($parser -notmatch 'parsed\s*<\s*0\.0' -or
    $parser -notmatch 'parsed\s*>=\s*360\.0') {
    throw 'camera rotation parser must accept only [0,360)'
}
if ($parser -match '360\s*\+\s*rounded') {
    throw 'positive-only protocol must not normalize negative angles'
}
if ($task -match 'TASK_SERVO_MAX_DEG|Task_SplitRotation') {
    throw 'positive R angles must no longer be split for the stepper'
}
Write-Output 'positive_angle_policy: PASS'

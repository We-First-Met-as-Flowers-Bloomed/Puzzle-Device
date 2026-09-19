$ErrorActionPreference = 'Stop'
$fourAxis = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw
$fourAxisHeader = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.h') -Raw
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw
$taskHeader = Get-Content (Join-Path $PSScriptRoot '..\app\task.h') -Raw
$main = Get-Content (Join-Path $PSScriptRoot '..\Core\Src\main.c') -Raw

foreach ($required in @(
    'FourAxis_SetTask1OpenLoop',
    'set_task1_open_loop',
    'task1_open_loop_mode',
    'generic_move_open_loop\[4\]',
    'software_homing_open_loop',
    'software_homing_open_loop_duration\[4\]',
    '\[TASK1 OPEN LOOP\] axis=%s START POSITION MISSING, SEND',
    '\[TASK1 OPEN LOOP\] axis=%s FD ACK MISSING, CONTINUE',
    '\[TASK1 OPEN LOOP\] axis=%s STATUS MISSING, TIME COMPLETE',
    '\[TASK1 OPEN LOOP\] axis=%s ARRIVAL TIMEOUT, CONTINUE',
    '\[TASK1 OPEN LOOP\] RETURN TIMEOUT, CONTINUE'
)) {
    if (($fourAxis -notmatch $required) -and ($fourAxisHeader -notmatch $required) -and
        ($task -notmatch $required) -and ($taskHeader -notmatch $required) -and
        ($main -notmatch $required)) {
        throw "missing Task 1 open-loop contract: $required"
    }
}

if ($fourAxis -notmatch 'EMM42_STATUS_STALLED[\s\S]*generic_move_failed\[axis\]\s*=\s*1u' -or
    $fourAxis -notmatch 'EMM42_STATUS_STALL_PROTECTION') {
    throw 'explicit stall and stall-protection must remain fatal'
}
if ($fourAxis -match 'TASK1 OPEN LOOP[\s\S]{0,800}Emm42_BuildPosition[\s\S]{0,400}Emm42_BuildPosition') {
    throw 'Task 1 open-loop fallback must not retransmit position commands'
}

Write-Output 'task1_open_loop_policy: PASS'

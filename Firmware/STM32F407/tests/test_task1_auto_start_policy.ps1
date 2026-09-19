$ErrorActionPreference = 'Stop'
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw
$header = Get-Content (Join-Path $PSScriptRoot '..\app\task.h') -Raw
$main = Get-Content (Join-Path $PSScriptRoot '..\Core\Src\main.c') -Raw

if ($header -notmatch 'void\s+Task_AutoStartTask1\s*\(\s*void\s*\)\s*;') {
    throw 'Task 1 automatic-start API is missing'
}
if ($task -notmatch 'void\s+Task_AutoStartTask1\s*\(\s*void\s*\)') {
    throw 'Task 1 automatic-start implementation is missing'
}
if ($task -notmatch 'void\s+Task_StartOrTogglePause\s*\([^)]*\)\s*\{\s*if\s*\(\s*current_task\s*==\s*1u\s*\)\s*return\s*;') {
    throw 'Task 1 must ignore START/PAUSE at the task layer'
}
if ($task -notmatch 'Task_SelectNext[\s\S]*Task_AutoStartTask1\s*\(\s*\)') {
    throw 'selecting Task 1 must automatically start it'
}

if ($main -notmatch 'FourAxis_GetStartupState\s*\(\s*\)\s*==\s*FOUR_AXIS_STARTUP_READY[\s\S]*?Task_AutoStartTask1\s*\(\s*\)[\s\S]*?VisionJsonl_ProcessWithRaw') {
    throw 'Task 1 must auto-start before the first camera JSON is processed'
}

Write-Output 'task1_auto_start_policy: PASS'

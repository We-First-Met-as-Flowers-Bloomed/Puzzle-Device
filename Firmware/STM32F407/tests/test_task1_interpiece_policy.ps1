$ErrorActionPreference = 'Stop'
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw
$fourAxis = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw

foreach ($required in @(
    '#define\s+TASK1_XY_SPEED_RPM\s+3000u',
    '#define\s+TASK1_Z_SPEED_RPM\s+1500u',
    '#define\s+TASK1_R_SPEED_RPM\s+3000u',
    '#define\s+TASK1_AXIS_ACCELERATION\s+50u',
    '#define\s+MAGNET_SETTLE_MS\s+500u',
    '#define\s+MAGNET_DEMAG_MS\s+50u',
    'task9_r_degrees',
    'queue_interpiece_pickup_group',
    'start_vision_piece\s*\(\s*task9_vision_index\s*\)'
)) {
    if (($task -notmatch $required) -and ($fourAxis -notmatch $required)) {
        throw "missing Task 1 inter-piece contract: $required"
    }
}

$task9 = [regex]::Match(
    $task,
    'static\s+void\s+task9_process\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $task9.Success) { throw 'missing Task 1/2 transfer state machine' }
if ($task9.Groups['body'].Value -notmatch 'task9_vision_index\s*<\s*task9_vision_plan\.count') {
    throw 'intermediate/final piece distinction missing'
}
if ($task9.Groups['body'].Value -notmatch 'current_task\s*==\s*1u') {
    throw 'direct transition must be isolated to Task 1'
}

Write-Output 'task1_interpiece_policy: PASS'

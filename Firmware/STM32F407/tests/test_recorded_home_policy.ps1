$ErrorActionPreference = 'Stop'
$source = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw
$header = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.h') -Raw

foreach ($required in @(
    'FourAxis_RecordedHomeStart',
    'FourAxis_RecordedHomeProcess',
    'FOUR_AXIS_RECORDED_HOME_RUNNING',
    '#define\s+RECORDED_HOME_TOLERANCE_PULSES\s+20u',
    '#define\s+RECORDED_HOME_MAX_CORRECTION_PULSES\s+500u',
    '#define\s+RECORDED_HOME_MAX_CORRECTION_ROUNDS\s+2u',
    'AxisPosition_SelectCorrection',
    'recorded_home_correction_round'
)) {
    if (($source -notmatch $required) -and ($header -notmatch $required)) {
        throw "missing recorded-home contract: $required"
    }
}

$start = [regex]::Match(
    $source,
    'void\s+FourAxis_RecordedHomeStart\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $start.Success -or
    $start.Groups['body'].Value -notmatch 'software_homing_start\s*\(\s*1u\s*\)') {
    throw 'Task 9 recorded home must begin with tracked-pulse coarse return'
}

if ($source -match 'recorded_home[\s\S]{0,500}Emm42_BuildSetHomingZero') {
    throw 'recorded home must not write driver flash'
}

Write-Output 'recorded_home_policy: PASS'

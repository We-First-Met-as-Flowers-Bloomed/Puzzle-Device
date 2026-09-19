$ErrorActionPreference = 'Stop'
$source = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw

if ($source -notmatch '#define\s+HOME_ENCODER_DEADBAND_PULSES\s+20u') {
    throw 'home encoder correction must use the approved 20-pulse deadband'
}
if ($source -notmatch '#define\s+HOME_ENCODER_MAX_CORRECTION_PULSES\s+500u') {
    throw 'home encoder correction must cap one attempt at 500 pulses'
}
if ($source -notmatch 'AxisPosition_SelectCorrection') {
    throw 'software homing must use the bounded correction selector'
}
if ($source -notmatch 'home_encoder_correction_start\s*\(' -or
    $source -notmatch 'home_encoder_correction_poll\s*\(') {
    throw 'software homing must contain one encoder correction phase'
}
$start = [regex]::Match(
    $source,
    'static\s+void\s+home_encoder_correction_start\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $start.Success) { throw 'encoder correction start function is missing' }
if ($start.Groups['body'].Value -match 'z_serial_axis') {
    throw 'Z must be excluded from automatic encoder correction'
}
if ($source -notmatch '\[HOME ENC WARN\]') {
    throw 'advisory encoder failures must be visible as warnings'
}
$poll = [regex]::Match(
    $source,
    'static\s+uint8_t\s+home_encoder_correction_poll\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $poll.Success) { throw 'encoder correction poll function is missing' }
if ($poll.Groups['body'].Value -match 'FOUR_AXIS_HOMING_ERROR') {
    throw 'encoder correction must never place homing into ERROR'
}

Write-Output 'home_encoder_correction_policy: PASS'

$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw

if ($source -notmatch '#define\s+X_MAX_PULSES\s+29500u') {
    throw 'X software coordinate maximum must be 29500 pulses'
}
if ($source -notmatch '#define\s+Y_MAX_PULSES\s+16800u') {
    throw 'Y software coordinate maximum must be 16800 pulses'
}
foreach ($axis in 'x','y','z') {
    if ($source -notmatch "AxisLimit_Init\s*\(\s*&$axis`_position_limit") {
        throw "$axis must initialize a software coordinate limiter at power-on"
    }
}
$homingStart = [regex]::Match(
    $source,
    'static\s+void\s+software_homing_start\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $homingStart.Success) {
    throw 'shared tracked-pulse software homing starter is missing'
}
$body = $homingStart.Groups['body'].Value
foreach ($axis in 'x','y','z','r') {
    if ($body -notmatch "&$axis`_serial_axis") {
        throw "task return-home must command $axis by tracked relative motion"
    }
}
if ($source -notmatch 'void\s+FourAxis_HomingStart\s*\(void\)[\s\S]*software_homing_start\s*\(\s*0u\s*\)') {
    throw 'ordinary task homing must enable the existing advisory correction path'
}

Write-Output 'software_home_policy: PASS'

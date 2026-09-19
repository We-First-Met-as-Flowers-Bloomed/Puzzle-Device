$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw

if ($source -notmatch '#define\s+X_SERIAL_DIRECTION_INVERT\s+0u') {
    throw 'X serial direction must be toggled after the physical axis swap'
}
if ($source -notmatch '#define\s+Y_SERIAL_DIRECTION_INVERT\s+0u') {
    throw 'Y serial direction must be toggled after the physical axis swap'
}
if ($source -notmatch 'axis\s*==\s*&x_serial_axis.*?X_SERIAL_DIRECTION_INVERT') {
    throw 'X inversion must be applied in the shared serial motion path'
}
if ($source -notmatch 'axis\s*==\s*&y_serial_axis.*?Y_SERIAL_DIRECTION_INVERT') {
    throw 'Y inversion must be applied in the shared serial motion path'
}

Write-Output 'serial_direction_policy: PASS'

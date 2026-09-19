$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw
$match = [regex]::Match(
    $source,
    'void\s+FourAxis_StartAll360\s*\(void\)\s*\{(?<body>.*?)\n\}',
    [System.Text.RegularExpressions.RegexOptions]::Singleline
)

if (-not $match.Success) {
    throw 'FourAxis_StartAll360 was not found'
}

$body = $match.Groups['body'].Value
if ($body -match 'serial_move_relative') {
    throw 'Task 1 startup must not pre-read serial-axis position'
}
if ($body -match 'FourAxis_AbortAll') {
    throw 'Task 1 startup must not abort healthy axes after one UART failure'
}
if ($body -notmatch 'serial_command_relative') {
    throw 'Task 1 startup must transmit UART motion commands directly'
}
if ($body -notmatch 'x_serial_axis' -or $body -notmatch 'y_serial_axis') {
    throw 'Task 1 must command both X and Y serial steppers'
}
if ($body -notmatch 'z_serial_axis') {
    throw 'Task 1 must command Z through the UART4 serial stepper'
}
if ($body -match 'htim10|htim11|PulseAxis') {
    throw 'Task 1 must not use pulse stepper outputs'
}

Write-Output 'motion_start_policy: PASS'

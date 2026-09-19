$source = Get-Content "$PSScriptRoot\..\hardware\four_axis.c" -Raw

if ($source -notmatch 'x_serial_axis\.uart\s*=\s*&huart6') {
    throw 'swapped long-travel X axis must use USART6 (PC6/PC7)'
}
if ($source -notmatch 'y_serial_axis\.uart\s*=\s*&huart3') {
    throw 'swapped short-travel Y axis must use USART3 (PD8/PD9)'
}
if ($source -notmatch 'z_serial_axis\.uart\s*=\s*&huart4') {
    throw 'Z axis must use UART4 (PA0/PC11)'
}
if ($source -notmatch 'r_serial_axis\.uart\s*=\s*&huart5') {
    throw 'R axis must use UART5 (PC12/PD2)'
}
if ($source -match 'htim10|htim11|pulse_axis|PulseAxis') {
    throw 'All linear axes are serial now; pulse stepper drivers must be gone from four_axis.c'
}

Write-Output 'serial_axis_mapping: PASS'

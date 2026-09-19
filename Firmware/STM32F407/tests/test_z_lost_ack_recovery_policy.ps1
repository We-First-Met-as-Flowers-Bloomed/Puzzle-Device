$ErrorActionPreference = 'Stop'
$source = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw
$positionSource = Get-Content (Join-Path $PSScriptRoot '..\hardware\axis_position.c') -Raw
$positionHeader = Get-Content (Join-Path $PSScriptRoot '..\hardware\axis_position.h') -Raw

foreach ($required in @(
    'SerialSendResult_t',
    'SERIAL_SEND_ACK_TIMEOUT',
    '#define\s+MOVE_ACK_RECOVERY_TIMEOUT_MS\s+300u',
    '#define\s+MOVE_POSITION_SNAPSHOT_ATTEMPTS\s+3u',
    'axis_recover_lost_move_ack',
    'axis_encoder_logical_positive_is_negative',
    'AxisPosition_HasMovedInDirection',
    '\[ACK RECOVER\] axis=%s FD ACK LOST',
    '\[ACK RECOVER\] axis=%s moved=%lu expected=%lu ACCEPT'
)) {
    if (($source -notmatch $required) -and
        ($positionSource -notmatch $required) -and
        ($positionHeader -notmatch $required)) {
        throw "missing Z lost-ACK recovery contract: $required"
    }
}

$recovery = [regex]::Match(
    $source,
    'static\s+uint8_t\s+axis_recover_lost_move_ack\s*\([^)]*\)\s*\{(?<body>[\s\S]*?)\n\}'
)
if (-not $recovery.Success) {
    throw 'missing bounded four-axis lost-ACK recovery function'
}
if ($recovery.Groups['body'].Value -match 'Emm42_BuildPosition|HAL_UART_Transmit\s*\([^;]*packet') {
    throw 'four-axis lost-ACK recovery must never retransmit the relative move'
}
if ($source -match 'z_recover_lost_move_ack') {
    throw 'Z-only recovery must be replaced by the generic four-axis path'
}
if ($source -notmatch 'send_result\s*==\s*SERIAL_SEND_ACK_TIMEOUT') {
    throw 'only an ACK timeout may enter four-axis recovery'
}
foreach ($axis in @('x_serial_axis', 'y_serial_axis', 'z_serial_axis', 'r_serial_axis')) {
    if ($source -notmatch [regex]::Escape($axis)) {
        throw "missing serial-axis coverage: $axis"
    }
}

Write-Output 'four_axis_lost_ack_recovery_policy: PASS'

$ErrorActionPreference = 'Stop'
$main = Get-Content (Join-Path $PSScriptRoot '..\Core\Src\main.c') -Raw
$axis = Get-Content (Join-Path $PSScriptRoot '..\hardware\four_axis.c') -Raw
$task = Get-Content (Join-Path $PSScriptRoot '..\app\task.c') -Raw
$project = Get-Content (Join-Path $PSScriptRoot '..\MDK-ARM\Automatic inspection vehicle.uvprojx') -Raw

if ($main -notmatch 'VisionJsonl_RxByte\(byte\)') {
    throw 'USART2 bytes must enter the JSONL receiver'
}
if ($main -match 'Task_CameraRxByte\(vision_rx_byte\)') {
    throw 'USART2 must no longer enter the legacy PIECE parser'
}
if ($main -notmatch 'VisionJsonl_Process') {
    throw 'the main loop must process complete JSONL lines'
}
if ($main -notmatch 'DebugUart_LogTextRx\("USART2",\s*"CAM",\s*vision_rx_log\)') {
    throw 'Task 1 must echo each complete USART2 JSON line on USART1'
}
if ($main -notmatch 'Task_SubmitVisionPlan') {
    throw 'validated JSONL plans must be submitted to Task 1'
}
if ($main -match 'HAL_UART_Transmit\(&huart2') {
    throw 'UART-JSONL-PULSE-v1 is receive-only; task replies must not be sent on USART2'
}
if ($axis -notmatch '#define\s+X_MAX_PULSES\s+29500u') {
    throw 'mechanical X limit must be 29500 pulses'
}
if ($axis -notmatch '#define\s+Y_MAX_PULSES\s+16800u') {
    throw 'mechanical Y limit must be 16800 pulses'
}
if ($task -notmatch '#define\s+TASK_CAMERA_X_MAX_PULSES\s+24000u' -or
    $task -notmatch '#define\s+TASK_CAMERA_Y_MAX_PULSES\s+16700u') {
    throw 'camera-domain X/Y maxima must remain 24000/16700'
}
if ($task -notmatch 'load_transfer_script\s*\(\s*\(uint16_t\)command->pickup_x_pulse\s*,\s*\(uint16_t\)command->pickup_y_pulse\s*,\s*\(uint16_t\)command->place_x_pulse\s*,\s*\(uint16_t\)command->place_y_pulse\s*,\s*command->rotation_deg\s*,\s*interpiece\s*\)') {
    throw 'Task 1 JSON commands must use the shared direct mechanical pulse execution'
}
if ($project -notmatch 'vision_jsonl\.c') {
    throw 'vision_jsonl.c must be included in the Keil project'
}

Write-Output 'serial_jsonl_policy: PASS'

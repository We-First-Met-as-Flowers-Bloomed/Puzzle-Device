$main = Get-Content "$PSScriptRoot\..\Core\Src\main.c" -Raw
$it = Get-Content "$PSScriptRoot\..\Core\Src\stm32f4xx_it.c" -Raw
$usart = Get-Content "$PSScriptRoot\..\Core\Src\usart.c" -Raw
$ioc = Get-Content "$PSScriptRoot\..\Automatic inspection vehicle.ioc" -Raw

if ($ioc -notmatch 'NVIC\.USART1_IRQn=true') {
    throw '.ioc must enable the USART1 NVIC line for the VOFA bridge'
}
if ($usart -notmatch 'HAL_NVIC_EnableIRQ\(USART1_IRQn\)') {
    throw 'USART1_IRQn must be enabled (VOFA command input)'
}
if ($it -notmatch 'void\s+USART1_IRQHandler\s*\(void\)') {
    throw 'USART1_IRQHandler must exist'
}
if ($main -notmatch 'HAL_UART_Receive_IT\(&huart1, &vofa_rx_byte, 1u\)') {
    throw 'main.c must arm interrupt receive on huart1'
}
if ($main -notmatch 'Task_CameraRxByte\(vofa_rx_byte\)') {
    throw 'USART1 bytes must be forwarded into the task command line parser'
}

Write-Output 'vofa_bridge_policy: PASS'

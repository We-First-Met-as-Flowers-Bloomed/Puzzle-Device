$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot '..'
$main=Get-Content -Raw (Join-Path $root 'Core\Src\main.c')
$axis=Get-Content -Raw (Join-Path $root 'hardware\four_axis.c')
if ($main -notmatch 'DebugUart_LogTextRx\("USART2",\s*"CAM"') { throw 'USART2 CAM mirror missing' }
if ($main -notmatch 'DebugUart_LogTextRx\("USART1",\s*"HOST"') { throw 'USART1 HOST mirror missing' }
if ($axis -notmatch 'DebugUart_LogFrameRx') { throw 'driver frame mirror missing' }
foreach ($label in @('"USART6"','"USART3"','"UART4"','"UART5"','return "X"','return "Y"','return "Z"','return "R"')) { if ($axis -notmatch [regex]::Escape($label)) { throw "driver label missing: $label" } }
if ($main -match 'HAL_UART_Receive_IT\s*\(\s*&huart(3|4|5|6)') { throw 'driver UART RX interrupts must not be introduced' }
Write-Output 'all_uart_mirror_policy: PASS'

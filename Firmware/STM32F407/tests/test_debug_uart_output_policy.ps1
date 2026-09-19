$ErrorActionPreference='Stop'
$main=Get-Content -Raw (Join-Path $PSScriptRoot '..\Core\Src\main.c')
if ($main -notmatch 'PUTCHAR_PROTOTYPE[\s\S]*DebugUart_PutChar') { throw 'stdio is not queued' }
if ($main -notmatch 'Task_SendResponse[\s\S]*DebugUart_Printf') { throw 'task response is not queued' }
if ($main -notmatch 'MX_USART1_UART_Init\s*\(\s*\)[\s\S]*DebugUart_Init\s*\(\s*\)') { throw 'debug UART init missing' }
if ($main -notmatch 'HAL_UART_TxCpltCallback[\s\S]*DebugUart_TxComplete') { throw 'DMA completion bridge missing' }
if ($main -match 'HAL_UART_Transmit\s*\(\s*&huart1') { throw 'blocking USART1 output remains' }
Write-Output 'debug_uart_output_policy: PASS'

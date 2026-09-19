$ErrorActionPreference='Stop'
$main=Get-Content -Raw (Join-Path $PSScriptRoot '..\Core\Src\main.c')
if($main -notmatch 'HAL_UARTEx_RxEventCallback'){throw 'USART2 RX event callback missing'}
if($main -notmatch 'HAL_UART_ErrorCallback'){throw 'USART2 error callback missing'}
if($main -notmatch 'usart2_recovery_pending[\s\S]*HAL_UART_AbortReceive[\s\S]*USART2_StartDmaRx' -or $main -notmatch 'USART2_StartDmaRx[\s\S]*HAL_UARTEx_ReceiveToIdle_DMA'){throw 'main-context USART2 restart missing'}
if($main -notmatch '\[USART2 ERROR\].*RECOVER' -or $main -notmatch '\[USART2 PARTIAL\].*NO NEWLINE'){throw 'USART2 diagnostics missing'}
$rxCallback=[regex]::Match($main,'void\s+HAL_UARTEx_RxEventCallback[\s\S]*?\n\}').Value
$errorCallback=[regex]::Match($main,'void\s+HAL_UART_ErrorCallback[\s\S]*?\n\}').Value
if($rxCallback -match 'VisionJsonl_RxByte|DebugUart_' -or $errorCallback -match 'DebugUart_'){throw 'callbacks must not parse or log'}
Write-Output 'usart2_recovery_policy: PASS'

$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot '..'
$ioc=Get-Content -Raw (Join-Path $root 'Automatic inspection vehicle.ioc')
$main=Get-Content -Raw (Join-Path $root 'Core\Src\main.c')
$usart=Get-Content -Raw (Join-Path $root 'Core\Src\usart.c')
$it=Get-Content -Raw (Join-Path $root 'Core\Src\stm32f4xx_it.c')
if($ioc -notmatch 'Dma\.Request\d+=USART2_RX' -or $ioc -notmatch 'Dma\.USART2_RX\.\d+\.Instance=DMA1_Stream5' -or $ioc -notmatch 'Dma\.USART2_RX\.\d+\.Mode=DMA_CIRCULAR'){throw 'USART2 circular RX DMA missing from .ioc'}
if($ioc -notmatch 'NVIC\.DMA1_Stream5_IRQn=true'){throw 'DMA1 Stream5 IRQ missing'}
if($usart -notmatch 'hdma_usart2_rx\.Instance\s*=\s*DMA1_Stream5' -or $usart -notmatch '__HAL_LINKDMA\(uartHandle,hdmarx,hdma_usart2_rx\)'){throw 'USART2 RX DMA handle/link missing'}
if($it -notmatch 'DMA1_Stream5_IRQHandler[\s\S]*HAL_DMA_IRQHandler\(&hdma_usart2_rx\)'){throw 'DMA1 Stream5 handler missing'}
if($main -notmatch 'HAL_UARTEx_ReceiveToIdle_DMA\s*\(\s*&huart2'){throw 'USART2 Receive-to-IDLE startup missing'}
if($main -match 'HAL_UART_Receive_IT\s*\(\s*&huart2'){throw 'one-byte USART2 interrupt reception must be removed'}
Write-Output 'usart2_rx_dma_policy: PASS'

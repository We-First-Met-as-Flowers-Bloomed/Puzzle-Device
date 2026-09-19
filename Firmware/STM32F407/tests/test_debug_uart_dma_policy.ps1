$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot '..'
$ioc = Get-Content -Raw (Join-Path $root 'Automatic inspection vehicle.ioc')
$main = Get-Content -Raw (Join-Path $root 'Core\Src\main.c')
$usart = Get-Content -Raw (Join-Path $root 'Core\Src\usart.c')
$it = Get-Content -Raw (Join-Path $root 'Core\Src\stm32f4xx_it.c')
if ($ioc -notmatch 'Dma\.Request\d+=USART1_TX' -or $ioc -notmatch 'Dma\.USART1_TX\.\d+\.Instance=DMA2_Stream7') { throw 'USART1 TX DMA2 Stream7 missing from .ioc' }
if ($ioc -notmatch 'NVIC\.DMA2_Stream7_IRQn=true') { throw 'DMA2 Stream7 NVIC missing' }
if ($main -notmatch 'MX_DMA_Init\s*\(\s*\)[\s\S]*MX_USART1_UART_Init') { throw 'DMA must initialize before USART1' }
if ($usart -notmatch 'hdma_usart1_tx\.Instance\s*=\s*DMA2_Stream7' -or $usart -notmatch '__HAL_LINKDMA\(uartHandle,hdmatx,hdma_usart1_tx\)') { throw 'USART1 TX DMA link missing' }
if ($it -notmatch 'DMA2_Stream7_IRQHandler[\s\S]*HAL_DMA_IRQHandler\(&hdma_usart1_tx\)') { throw 'DMA IRQ handler missing' }
if ($main -match 'HAL_UART_Transmit\s*\(\s*&huart1') { throw 'Blocking USART1 transmit remains in main.c' }
Write-Output 'debug_uart_dma_policy: PASS'

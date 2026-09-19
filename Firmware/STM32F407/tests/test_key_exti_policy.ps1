$ioc = Get-Content "$PSScriptRoot\..\Automatic inspection vehicle.ioc" -Raw
$gpio = Get-Content "$PSScriptRoot\..\Core\Src\gpio.c" -Raw
$it = Get-Content "$PSScriptRoot\..\Core\Src\stm32f4xx_it.c" -Raw
$main = Get-Content "$PSScriptRoot\..\Core\Src\main.c" -Raw

foreach ($pin in 12..15) {
    if ($ioc -notmatch "PB$pin\.GPIO_ModeDefaultEXTI=GPIO_MODE_IT_FALLING") {
        throw "PB$pin must use falling-edge EXTI in the .ioc"
    }
    if ($ioc -notmatch "PB$pin\.GPIO_PuPd=GPIO_PULLUP") {
        throw "PB$pin must use an internal pull-up in the .ioc"
    }
    if ($ioc -notmatch "PB$pin\.Signal=GPXTI$pin") {
        throw "PB$pin must be routed to EXTI$pin in the .ioc"
    }
}
if ($ioc -notmatch 'NVIC\.EXTI15_10_IRQn=true') {
    throw 'EXTI15_10_IRQn must be enabled in the .ioc'
}
if ($gpio -notmatch 'key1_Pin\|key2_Pin\|key3_Pin\|key4_Pin[\s\S]*GPIO_MODE_IT_FALLING[\s\S]*GPIO_PULLUP') {
    throw 'generated GPIO initialization must configure all keys as falling-edge pull-up inputs'
}
if ($gpio -notmatch 'HAL_NVIC_EnableIRQ\(EXTI15_10_IRQn\)') {
    throw 'generated GPIO initialization must enable EXTI15_10_IRQn'
}
if ($gpio -match 'KeyExti_Restore' -or $main -match 'KeyExti_Restore') {
    throw 'legacy duplicate KeyExti_Restore patch must be removed'
}
$handlerMatches = [regex]::Matches($it, 'void\s+EXTI15_10_IRQHandler\s*\(void\)')
if ($handlerMatches.Count -ne 1) {
    throw 'exactly one EXTI15_10_IRQHandler must serve all four keys'
}
foreach ($key in 1..4) {
    if ($it -notmatch "HAL_GPIO_EXTI_IRQHandler\(key$key`_Pin\)") {
        throw "IRQ handler must dispatch key$key"
    }
}
if ($main -notmatch 'HAL_GPIO_ReadPin\(key1_GPIO_Port,\s*key1_Pin\)\s*==\s*GPIO_PIN_RESET') {
    throw 'main loop must interpret low key level as pressed'
}
if ($main -notmatch 'HAL_GPIO_EXTI_Callback[\s\S]*KeyInput_RecordIsr') {
    throw 'HAL EXTI callback must feed the key event queue'
}

Write-Output 'key_exti_policy: PASS'

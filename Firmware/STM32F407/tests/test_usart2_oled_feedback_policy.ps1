$ErrorActionPreference = 'Stop'
$main = Get-Content (Join-Path $PSScriptRoot '..\Core\Src\main.c') -Raw
$ui = Get-Content (Join-Path $PSScriptRoot '..\hardware\status_ui.c') -Raw
$header = Get-Content (Join-Path $PSScriptRoot '..\hardware\status_ui.h') -Raw

if ($header -notmatch 'void\s+StatusUi_ShowTransient\s*\(\s*const\s+char\s*\*\s*message\s*\)') {
    throw 'status_ui must expose a transient OLED message API'
}
if ($ui -notmatch '1000u' -or $ui -notmatch 'transient_message') {
    throw 'transient OLED messages must expire non-blockingly after one second'
}
if ($ui -notmatch 'OLED_ShowString\s*\(\s*0\s*,\s*6\s*,[^;]*transient_message') {
    throw 'the transient message must render on the bottom OLED line'
}
if ($ui -notmatch '"TASK: %u/9"') {
    throw 'OLED task counter must expose exactly Tasks 1 through 9'
}
foreach ($message in @('UART2 RX', 'JSON ACCEPT', 'JSON INVALID', 'TASK BUSY', 'UART2 REPORT')) {
    if ($main -notmatch [regex]::Escape($message)) {
        throw "main loop must publish OLED status: $message"
    }
}

Write-Output 'usart2_oled_feedback_policy: PASS'

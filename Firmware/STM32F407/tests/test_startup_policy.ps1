$main = Get-Content "$PSScriptRoot\..\Core\Src\main.c" -Raw
$ui = Get-Content "$PSScriptRoot\..\hardware\status_ui.c" -Raw

if ($main -match 'FourAxis_Zero(AllSteppers|Axis)\s*\(') {
    throw 'startup must not call any stepper zero command'
}

if ($main -notmatch 'FourAxis_Process\s*\(\s*\)\s*;[\s\S]*FourAxis_GetStartupState\s*\(\s*\)\s*==\s*FOUR_AXIS_STARTUP_READY') {
    throw 'main must process startup continuously and gate task input until ready'
}
if ($ui -notmatch 'FourAxis_GetStartupText') {
    throw 'OLED must display the Emm42 startup phase or error'
}

Write-Output 'startup_policy: PASS'

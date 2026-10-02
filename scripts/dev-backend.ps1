# Run the Stellaria backend in --dev mode (port 8765, token "dev", CORS *) against a dev
# library in stellaria\.dev\. Pass -Fake (or --fake) to simulate the engine.
#
#   scripts\dev-backend.ps1 [-Fake] [extra backend args...]
#
# Python: $env:STELLARIA_PYTHON, else the active virtualenv, else python.
param(
    [switch]$Fake,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = 'Stop'

$Stellaria = Split-Path -Parent $PSScriptRoot
$Engine = Join-Path $Stellaria 'engine'
if ($env:STELLARIA_PYTHON) { $Python = $env:STELLARIA_PYTHON }
elseif ($env:VIRTUAL_ENV) { $Python = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
else { $Python = 'python' }

$Library = Join-Path $Stellaria '.dev\library'
$Data = Join-Path $Stellaria '.dev\data'
New-Item -ItemType Directory -Force -Path $Library, $Data | Out-Null

$cli = @('-m', 'stellaria_backend', '--dev', '--library', $Library, '--data', $Data,
         '--engine', $Engine, '--resources', (Join-Path $Stellaria 'resources\manifest.json'))
if ($Fake) { $cli += '--fake-engine' }
foreach ($a in $Rest) { if ($a -eq '--fake') { $cli += '--fake-engine' } else { $cli += $a } }

$env:PYTHONUTF8 = '1'
Push-Location (Join-Path $Stellaria 'backend')
try { & $Python @cli } finally { Pop-Location }

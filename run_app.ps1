param(
  [int]$ApiPort = 8000,
  [int]$WebPort = 3000,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WebDir = Join-Path $Root "apps\web"
$LogDir = Join-Path $Root ".runtime_logs"
$ApiOut = Join-Path $LogDir "api.out.log"
$ApiErr = Join-Path $LogDir "api.err.log"
$WebOut = Join-Path $LogDir "web.out.log"
$WebErr = Join-Path $LogDir "web.err.log"

function Write-Host {
  param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [object[]]$Object,
    [ConsoleColor]$ForegroundColor,
    [switch]$NoNewline
  )
  try {
    if ($PSBoundParameters.ContainsKey("ForegroundColor")) {
      if ($NoNewline) {
        Microsoft.PowerShell.Utility\Write-Host @Object -ForegroundColor $ForegroundColor -NoNewline
      } else {
        Microsoft.PowerShell.Utility\Write-Host @Object -ForegroundColor $ForegroundColor
      }
    } else {
      if ($NoNewline) {
        Microsoft.PowerShell.Utility\Write-Host @Object -NoNewline
      } else {
        Microsoft.PowerShell.Utility\Write-Host @Object
      }
    }
  } catch {
    try {
      [Console]::Out.WriteLine(($Object -join " "))
    } catch {
      # Ignore broken stdout pipes during automated shutdown.
    }
  }
}

function Write-Step($Message) {
  Write-Host ""
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command($Name, $InstallHint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Missing required command '$Name'. $InstallHint"
  }
}

function Stop-ProcessTree([int]$ProcessId) {
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
  foreach ($child in $children) {
    Stop-ProcessTree -ProcessId $child.ProcessId
  }
  $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if ($process) {
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
  }
}

function Test-PythonReady {
  Push-Location $Root
  try {
    & python -c "import fastapi, uvicorn, reliaguard_studio" *> $null
    return ($LASTEXITCODE -eq 0)
  } finally {
    Pop-Location
  }
}

function Ensure-Dependencies {
  if ($SkipInstall) {
    Write-Step "Skipping dependency installation checks"
    return
  }

  Write-Step "Checking Python package dependencies"
  if (-not (Test-PythonReady)) {
    Write-Host "Installing Python package in editable mode..." -ForegroundColor Yellow
    Push-Location $Root
    try {
      & python -m pip install -e ".[dev]"
      if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
    } finally {
      Pop-Location
    }
  } else {
    Write-Host "Python dependencies look ready." -ForegroundColor Green
  }

  Write-Step "Checking frontend dependencies"
  if (-not (Test-Path (Join-Path $WebDir "node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location $WebDir
    try {
      & npm install --no-audit --no-fund
      if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    } finally {
      Pop-Location
    }
  } else {
    Write-Host "Frontend dependencies look ready." -ForegroundColor Green
  }
}

function Start-LoggedProcess($Name, $FilePath, $Arguments, $WorkingDirectory, $StdOut, $StdErr) {
  if (Test-Path $StdOut) { Remove-Item $StdOut -Force }
  if (Test-Path $StdErr) { Remove-Item $StdErr -Force }
  Write-Step "Starting $Name"
  $process = Start-Process `
    -FilePath $FilePath `
    -ArgumentList $Arguments `
    -WorkingDirectory $WorkingDirectory `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr `
    -NoNewWindow `
    -PassThru
  Write-Host "$Name PID: $($process.Id)" -ForegroundColor DarkGray
  return $process
}

function Read-NewLogLines($Path, [ref]$Offset, $Prefix, $Color) {
  if (-not (Test-Path $Path)) { return }
  $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
  try {
    if ($Offset.Value -gt $stream.Length) { $Offset.Value = 0 }
    $stream.Seek($Offset.Value, [System.IO.SeekOrigin]::Begin) | Out-Null
    $reader = New-Object System.IO.StreamReader($stream)
    while (-not $reader.EndOfStream) {
      $line = $reader.ReadLine()
      if ($line) {
        Write-Host "[$Prefix] $line" -ForegroundColor $Color
      }
    }
    $Offset.Value = $stream.Position
  } finally {
    $stream.Close()
  }
}

Require-Command "python" "Install Python 3.11+ and rerun this script."
Require-Command "npm" "Install Node.js/npm and rerun this script."

if (-not (Test-Path $WebDir)) {
  throw "Could not find apps\web. Run this script from the ReliaGuard Studio repository root."
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONPATH = "$Root\src;$Root;$env:PYTHONPATH"
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:$ApiPort"

Ensure-Dependencies

$api = $null
$web = $null
$apiOutOffset = [ref]0L
$apiErrOffset = [ref]0L
$webOutOffset = [ref]0L
$webErrOffset = [ref]0L

try {
  $apiArgs = @("-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort")
  $webArgs = @("/d", "/s", "/c", "npm run dev -- --hostname 127.0.0.1 --port $WebPort")

  $api = Start-LoggedProcess -Name "ReliaGuard API" -FilePath "python" -Arguments $apiArgs -WorkingDirectory $Root -StdOut $ApiOut -StdErr $ApiErr
  Start-Sleep -Seconds 2
  $web = Start-LoggedProcess -Name "ReliaGuard Web" -FilePath "cmd.exe" -Arguments $webArgs -WorkingDirectory $WebDir -StdOut $WebOut -StdErr $WebErr

  Write-Host ""
  Write-Host "ReliaGuard Studio is starting." -ForegroundColor Green
  Write-Host "API: http://127.0.0.1:$ApiPort" -ForegroundColor Green
  Write-Host "Web: http://127.0.0.1:$WebPort" -ForegroundColor Green
  Write-Host ""
  Write-Host "This launcher does not open a browser automatically. Open the Web URL manually." -ForegroundColor Yellow
  Write-Host "Press Ctrl+C to stop both services." -ForegroundColor Yellow

  while ($true) {
    if ($api.HasExited) { throw "ReliaGuard API stopped unexpectedly. See $ApiErr" }
    if ($web.HasExited) { throw "ReliaGuard Web stopped unexpectedly. See $WebErr" }

    Read-NewLogLines -Path $ApiOut -Offset $apiOutOffset -Prefix "api" -Color Gray
    Read-NewLogLines -Path $ApiErr -Offset $apiErrOffset -Prefix "api!" -Color DarkYellow
    Read-NewLogLines -Path $WebOut -Offset $webOutOffset -Prefix "web" -Color Gray
    Read-NewLogLines -Path $WebErr -Offset $webErrOffset -Prefix "web!" -Color DarkYellow

    Start-Sleep -Milliseconds 750
  }
} finally {
  Write-Host ""
  Write-Step "Stopping ReliaGuard Studio"
  if ($web -and -not $web.HasExited) { Stop-ProcessTree -ProcessId $web.Id }
  if ($api -and -not $api.HasExited) { Stop-ProcessTree -ProcessId $api.Id }
  Write-Host "Stopped API and web processes." -ForegroundColor Green
}

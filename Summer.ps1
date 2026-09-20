Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Core = Join-Path $Root "summer.py"

if (-not (Test-Path -LiteralPath $Core -PathType Leaf)) {
    Write-Error "summer.py was not found next to Summer.ps1: $Core"
    exit 2
}

$Python = $null
$PythonMode = $null

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PyLauncher) {
    $Python = $PyLauncher.Source
    $PythonMode = "py"
} else {
    $Python3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($Python3) {
        $Python = $Python3.Source
        $PythonMode = "python"
    } else {
        $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($PythonCmd) {
            $Python = $PythonCmd.Source
            $PythonMode = "python"
        }
    }
}

if (-not $Python) {
    Write-Error "Python 3 was not found. Install Python 3, then run Summer.ps1 again."
    exit 127
}

$LaunchArgs = @($args)

if ($LaunchArgs.Count -eq 0) {
    Write-Host ""
    Write-Host "Summer - Qwen3.5 MoE / llama.cpp"
    Write-Host "Interactive Windows setup. Press Enter to accept defaults."
    Write-Host ""

    $Mode = Read-Host "Mode [cli]"
    if ([string]::IsNullOrWhiteSpace($Mode)) { $Mode = "cli" }
    if ($Mode -notin @("cli", "server")) {
        Write-Error "Mode must be 'cli' or 'server'."
        exit 2
    }

    $DefaultExe = if ($Mode -eq "server") { "llama-server.exe" } else { "llama-cli.exe" }
    $Bin = Read-Host "Path to $DefaultExe"
    if ([string]::IsNullOrWhiteSpace($Bin)) {
        Write-Error "A llama.cpp executable path is required."
        exit 2
    }

    $Model = Read-Host "Path to Qwen3.5 MoE GGUF"
    if ([string]::IsNullOrWhiteSpace($Model)) {
        Write-Error "A GGUF model path is required."
        exit 2
    }

    $Profile = Read-Host "Profile: balanced / low-vram / gpu / long-context [balanced]"
    if ([string]::IsNullOrWhiteSpace($Profile)) { $Profile = "balanced" }
    if ($Profile -notin @("balanced", "low-vram", "gpu", "long-context")) {
        Write-Error "Unknown profile: $Profile"
        exit 2
    }

    $LaunchArgs = @(
        "--mode", $Mode,
        "--bin", $Bin,
        "--model", $Model,
        "--profile", $Profile
    )
}

if ($PythonMode -eq "py") {
    & $Python -3 $Core @LaunchArgs
} else {
    & $Python $Core @LaunchArgs
}

exit $LASTEXITCODE

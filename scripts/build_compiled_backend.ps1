param(
    [string]$PythonExe = "",
    [string]$OutputRoot = "",
    [switch]$KeepTemp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$RequestedPath)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $candidates = New-Object System.Collections.Generic.List[string]
    $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $bundled = Join-Path $repoRoot "runtime\python\python.exe"
    if (Test-Path -LiteralPath $bundled) {
        $candidates.Add($bundled)
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython) {
            $candidates.Add($condaPython)
        }
    }

    foreach ($candidate in $candidates) {
        try {
            & $candidate -c "import Cython" > $null 2> $null
        } catch {
            $global:LASTEXITCODE = 1
        }
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        try {
            & $command.Source -c "import Cython" > $null 2> $null
        } catch {
            $global:LASTEXITCODE = 1
        }
        if ($LASTEXITCODE -eq 0) {
            return $command.Source
        }
    }

    if ($candidates.Count -gt 0) {
        return $candidates[0]
    }

    throw "Python executable not found. Pass -PythonExe explicitly."
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Resolve-PythonExe -RequestedPath $PythonExe
$scriptPath = Join-Path $repoRoot "scripts\build_compiled_backend.py"
$resolvedOutputRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    Join-Path $repoRoot "build\compiled-backend"
} else {
    if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
        [System.IO.Path]::GetFullPath($OutputRoot)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
    }
}

Write-Host ""
Write-Host "==> Building compiled backend bundle" -ForegroundColor Cyan
Write-Host "Python: $python"
Write-Host "Output: $resolvedOutputRoot"

Invoke-Checked -Executable $python -Arguments @("-c", "import Cython")

$buildArgs = @($scriptPath, "--output-root", $resolvedOutputRoot)
if ($KeepTemp) {
    $buildArgs += "--keep-temp"
}
Invoke-Checked -Executable $python -Arguments $buildArgs

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $resolvedOutputRoot
    Invoke-Checked -Executable $python -Arguments @("-m", "realtime.app", "--help")
} finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

Write-Host ""
Write-Host "Compiled backend bundle validation passed." -ForegroundColor Green

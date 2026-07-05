param(
    [string]$EnvName = "",
    [string]$ArchivePath = "",
    [string]$TargetDir = "",
    [switch]$ReplaceExisting,
    [switch]$SkipValidation,
    [switch]$StrictEditableCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Required command not found: $Name"
    }
    return $command
}

function Get-AbsolutePath {
    param(
        [string]$BaseDir,
        [string]$Candidate
    )

    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        return [System.IO.Path]::GetFullPath($Candidate)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Candidate))
}

function Assert-SafeSubpath {
    param(
        [string]$RootPath,
        [string]$ChildPath
    )

    $root = [System.IO.Path]::GetFullPath($RootPath)
    $child = [System.IO.Path]::GetFullPath($ChildPath)
    $withSlash = if ($root.EndsWith([System.IO.Path]::DirectorySeparatorChar)) { $root } else { "$root\" }
    if (-not $child.StartsWith($withSlash, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside repo root. Root: $root Child: $child"
    }
}

function Get-MeaningfulItems {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }

    $ignoredNames = @(".gitkeep", "README.md")
    return @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue | Where-Object {
        $ignoredNames -notcontains $_.Name
    })
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$runtimeRoot = Join-Path $repoRoot "runtime"

if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
    $ArchivePath = Join-Path $runtimeRoot "$EnvName.zip"
} else {
    $ArchivePath = Get-AbsolutePath -BaseDir $repoRoot -Candidate $ArchivePath
}

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = Join-Path $runtimeRoot "python"
} else {
    $TargetDir = Get-AbsolutePath -BaseDir $repoRoot -Candidate $TargetDir
}

Assert-SafeSubpath -RootPath $repoRoot -ChildPath $ArchivePath
Assert-SafeSubpath -RootPath $repoRoot -ChildPath $TargetDir

$condaCommand = Require-Command -Name "conda"
$condaPackCommand = Require-Command -Name "conda-pack"

Write-Step "Repo root"
Write-Host $repoRoot

$envListJson = & $condaCommand.Source env list --json
$envList = $envListJson | ConvertFrom-Json
$envPath = $null
$resolvedEnvName = $EnvName

if (-not [string]::IsNullOrWhiteSpace($resolvedEnvName)) {
    Write-Step "Checking conda environment '$resolvedEnvName'"
    foreach ($prefix in $envList.envs) {
        if ([System.IO.Path]::GetFileName($prefix) -ieq $resolvedEnvName) {
            $envPath = $prefix
            break
        }
    }
} elseif ($env:CONDA_PREFIX -and (Test-Path -LiteralPath $env:CONDA_PREFIX)) {
    $envPath = [System.IO.Path]::GetFullPath($env:CONDA_PREFIX)
    $resolvedEnvName = [System.IO.Path]::GetFileName($envPath)
    Write-Step "Using active conda environment '$resolvedEnvName'"
} elseif ($env:CONDA_DEFAULT_ENV) {
    $resolvedEnvName = $env:CONDA_DEFAULT_ENV
    Write-Step "Checking conda environment '$resolvedEnvName'"
    foreach ($prefix in $envList.envs) {
        if ([System.IO.Path]::GetFileName($prefix) -ieq $resolvedEnvName) {
            $envPath = $prefix
            break
        }
    }
}

if (-not $envPath) {
    throw "Conda environment not found. Activate the target environment first or pass -EnvName explicitly."
}

Write-Host "Found: $envPath" -ForegroundColor Green

Write-Step "Preparing runtime directories"
if (-not (Test-Path -LiteralPath $runtimeRoot)) {
    New-Item -ItemType Directory -Path $runtimeRoot | Out-Null
}

if (Test-Path -LiteralPath $TargetDir) {
    $existingItems = Get-MeaningfulItems -Path $TargetDir
    if ($existingItems -and -not $ReplaceExisting) {
        throw "Target directory already contains files: $TargetDir . Re-run with -ReplaceExisting to overwrite it."
    }

    if ($existingItems) {
        Write-Step "Removing existing runtime directory"
        Remove-Item -LiteralPath $TargetDir -Recurse -Force
    }
} else {
    $parentDir = Split-Path -Parent $TargetDir
    if (-not (Test-Path -LiteralPath $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir | Out-Null
    }
}

if (Test-Path -LiteralPath $ArchivePath) {
    Write-Step "Removing old archive"
    Remove-Item -LiteralPath $ArchivePath -Force
}

Write-Step "Packing conda environment"
$packArgs = @("-p", $envPath, "-o", $ArchivePath, "--format", "zip")
if (-not $StrictEditableCheck) {
    $packArgs += "--ignore-editable-packages"
    Write-Host "Ignoring editable package checks because repo source folders are bundled separately." -ForegroundColor Yellow
}

& $condaPackCommand.Source @packArgs
if ($LASTEXITCODE -ne 0) {
    throw "conda-pack failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    throw "conda-pack did not produce the expected archive: $ArchivePath"
}

Write-Step "Extracting archive"
Expand-Archive -LiteralPath $ArchivePath -DestinationPath $TargetDir -Force

$condaUnpack = Join-Path $TargetDir "Scripts\conda-unpack.exe"
if (Test-Path -LiteralPath $condaUnpack) {
    Write-Step "Running conda-unpack"
    & $condaUnpack
    if ($LASTEXITCODE -ne 0) {
        throw "conda-unpack failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "conda-unpack.exe not found, skipping relocation fixups." -ForegroundColor Yellow
}

$pythonExe = Join-Path $TargetDir "python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Bundled runtime is missing python.exe: $pythonExe"
}

if (-not $SkipValidation) {
    Write-Step "Validating bundled runtime"
    & $pythonExe --version
    & $pythonExe -c "import sys; print(sys.executable)"
    & $pythonExe -m realtime.app --help | Out-Null
    Write-Host "Runtime validation passed." -ForegroundColor Green
}

$archiveInfo = Get-Item -LiteralPath $ArchivePath

Write-Step "Done"
Write-Host "Archive: $ArchivePath"
Write-Host ("Archive size: {0:N2} GB" -f ($archiveInfo.Length / 1GB))
Write-Host "Runtime dir: $TargetDir"
Write-Host "Python: $pythonExe"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. cd electron-app"
Write-Host "2. npm install"
Write-Host "3. npm run dist"

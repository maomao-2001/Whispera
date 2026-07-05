param(
    [string]$RuntimeRoot = "runtime\python",
    [switch]$Aggressive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-AbsolutePath {
    param(
        [string]$BaseDir,
        [string]$Candidate
    )

    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        return [System.IO.Path]::GetFullPath($Candidate)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Candidate))
}

function Get-DirectorySizeBytes {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }

    return (Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
}

function Remove-ByFilter {
    param(
        [string]$Root,
        [string]$Filter
    )

    $files = @(Get-ChildItem -LiteralPath $Root -Recurse -File -Filter $Filter -ErrorAction SilentlyContinue)
    if (-not $files.Count) {
        return [PSCustomObject]@{ Pattern = $Filter; Count = 0; SizeBytes = 0 }
    }

    $sum = ($files | Measure-Object Length -Sum).Sum
    foreach ($file in $files) {
        Remove-Item -LiteralPath $file.FullName -Force
    }

    return [PSCustomObject]@{ Pattern = $Filter; Count = $files.Count; SizeBytes = $sum }
}

function Remove-DirectoriesByName {
    param(
        [string]$Root,
        [string[]]$Names
    )

    $dirs = @(Get-ChildItem -LiteralPath $Root -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object {
        $Names -contains $_.Name
    })

    if (-not $dirs.Count) {
        return [PSCustomObject]@{ Kind = "named_dirs"; Count = 0; SizeBytes = 0 }
    }

    $sum = 0
    foreach ($dir in $dirs) {
        $sum += Get-DirectorySizeBytes -Path $dir.FullName
    }

    foreach ($dir in $dirs | Sort-Object FullName -Descending) {
        Remove-Item -LiteralPath $dir.FullName -Recurse -Force
    }

    return [PSCustomObject]@{ Kind = "named_dirs"; Count = $dirs.Count; SizeBytes = $sum }
}

function Remove-PathsIfPresent {
    param([string[]]$Paths)

    $removed = @()
    foreach ($target in $Paths) {
        if (-not (Test-Path -LiteralPath $target)) {
            continue
        }
        $size = if ((Get-Item -LiteralPath $target).PSIsContainer) { Get-DirectorySizeBytes -Path $target } else { (Get-Item -LiteralPath $target).Length }
        Remove-Item -LiteralPath $target -Recurse -Force
        $removed += [PSCustomObject]@{ Path = $target; SizeBytes = $size }
    }
    return $removed
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".."))
$runtimePath = Get-AbsolutePath -BaseDir $repoRoot -Candidate $RuntimeRoot

if (-not (Test-Path -LiteralPath $runtimePath)) {
    throw "Runtime path not found: $runtimePath"
}

$beforeBytes = Get-DirectorySizeBytes -Path $runtimePath
Write-Step "Runtime before slimming"
Write-Host ("Path: {0}" -f $runtimePath)
Write-Host ("Size: {0:N2} GB" -f ($beforeBytes / 1GB))

$results = @()

Write-Step "Removing debug and build artifacts"
$results += Remove-ByFilter -Root $runtimePath -Filter "*.pdb"
$results += Remove-ByFilter -Root $runtimePath -Filter "*.lib"
$results += Remove-ByFilter -Root $runtimePath -Filter "*.map"
$results += Remove-ByFilter -Root $runtimePath -Filter "*.pyc"

Write-Step "Removing caches, tests, and docs"
$results += Remove-DirectoriesByName -Root $runtimePath -Names @("__pycache__", "tests", "test", "testing", "docs", "doc")

Write-Step "Removing package headers and static assets not needed at runtime"
$specificTargets = @(
    (Join-Path $runtimePath "Lib\site-packages\torch\include"),
    (Join-Path $runtimePath "Lib\site-packages\torch\share"),
    (Join-Path $runtimePath "Scripts\ruff.exe")
)

$packageTargets = @(
    "Lib\site-packages\gradio",
    "Lib\site-packages\gradio-5.49.1.dist-info",
    "Lib\site-packages\gradio_client",
    "Lib\site-packages\gradio_client-1.13.3.dist-info",
    "Lib\site-packages\pandas",
    "Lib\site-packages\pandas.libs",
    "Lib\site-packages\pandas-2.3.3.dist-info",
    "Lib\site-packages\pyarrow",
    "Lib\site-packages\pyarrow.libs",
    "Lib\site-packages\pyarrow-23.0.1.dist-info",
    "Lib\site-packages\ruff",
    "Lib\site-packages\ruff-0.14.2.dist-info"
)

if ($Aggressive) {
    $packageTargets += @(
        "Lib\site-packages\sklearn",
        "Lib\site-packages\scikit_learn-1.7.2.dist-info"
    )
}

$removedTargets = Remove-PathsIfPresent -Paths ($specificTargets + ($packageTargets | ForEach-Object { Join-Path $runtimePath $_ }))

$afterBytes = Get-DirectorySizeBytes -Path $runtimePath

Write-Step "Summary"
foreach ($result in $results) {
    if ($result.Count -gt 0) {
        $label = if ($result.PSObject.Properties["Pattern"]) { $result.Pattern } else { $result.Kind }
        Write-Host ("{0}: removed {1} items, saved {2:N2} GB" -f $label, $result.Count, ($result.SizeBytes / 1GB))
    }
}

foreach ($target in $removedTargets) {
    Write-Host ("removed path: {0} ({1:N2} GB)" -f $target.Path, ($target.SizeBytes / 1GB))
}

Write-Host ("Before: {0:N2} GB" -f ($beforeBytes / 1GB))
Write-Host ("After : {0:N2} GB" -f ($afterBytes / 1GB))
Write-Host ("Saved : {0:N2} GB" -f (($beforeBytes - $afterBytes) / 1GB))

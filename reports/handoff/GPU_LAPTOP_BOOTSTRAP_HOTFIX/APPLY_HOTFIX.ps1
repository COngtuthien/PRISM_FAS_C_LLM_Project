<#
    Applies the PRISM-FAS-C bootstrap hotfix to a deployed project folder.

    Copies only the runtime files listed in HOTFIX_MANIFEST.json, verifies every
    SHA256 after the copy, and stops on the first mismatch. It never touches
    data/, weights/, assets/, runs/, reports/ or state/, never deletes anything,
    and never starts training.

    Usage:
        powershell -ExecutionPolicy Bypass -File APPLY_HOTFIX.ps1 `
            -ProjectRoot "C:\path\to\PRISM_FAS_C_LLM_Project"
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"
$package = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $package "HOTFIX_MANIFEST.json"

if (-not (Test-Path $manifestPath)) {
    Write-Error "HOTFIX_MANIFEST.json is missing next to this script."
}
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Default only to a project root that is unmistakably one.
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $candidate = (Resolve-Path (Join-Path $package "..\..\..")).Path
    $looksRight = (Test-Path (Join-Path $candidate "train.py")) -and
                  (Test-Path (Join-Path $candidate "bootstrap.py")) -and
                  (Test-Path (Join-Path $candidate "configs\environment\environment_contract.yaml"))
    if (-not $looksRight) {
        Write-Error "Pass -ProjectRoot explicitly: the folder above this package does not look like a project root."
    }
    $ProjectRoot = $candidate
    Write-Host "ProjectRoot not given; using the enclosing project at $ProjectRoot"
}

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
foreach ($required in @("train.py", "bootstrap.py", "configs\environment\environment_contract.yaml")) {
    if (-not (Test-Path (Join-Path $ProjectRoot $required))) {
        Write-Error "$ProjectRoot does not look like a PRISM-FAS-C project (missing $required). Nothing was copied."
    }
}

# Paths this script must never write to, whatever a manifest says.
$forbidden = @("data", "weights", "assets", "runs", "reports", "state", ".git", ".venv")

$verified = 0
$total = ($manifest.files | Measure-Object).Count
foreach ($file in $manifest.files) {
    $relative = $file.destination -replace "/", "\"
    $top = ($relative -split "\\")[0]
    if ($forbidden -contains $top) {
        Write-Error "refusing to write into $top ($relative). Nothing further was copied."
    }

    $source = Join-Path $package $relative
    $destination = Join-Path $ProjectRoot $relative
    if (-not (Test-Path $source)) {
        Write-Error "the package is incomplete: $relative is missing."
    }

    if ($WhatIfOnly) {
        Write-Host ("WOULD COPY  {0}" -f $relative)
        continue
    }

    $parent = Split-Path -Parent $destination
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Copy-Item -Path $source -Destination $destination -Force

    $actual = (Get-FileHash -Path $destination -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $file.sha256) {
        Write-Error "SHA256 mismatch after copying $relative (got $actual, expected $($file.sha256))."
    }
    $verified += 1
    Write-Host ("OK  {0}  {1}" -f $file.sha256.Substring(0, 12), $relative)
}

if ($WhatIfOnly) {
    Write-Host "WHAT-IF only: nothing was written."
    exit 0
}

Write-Host ""
Write-Host "HOTFIX_APPLIED = PASS"
Write-Host ("FILES_VERIFIED = {0}/{1}" -f $verified, $total)
Write-Host "NEXT_COMMAND = python train.py"

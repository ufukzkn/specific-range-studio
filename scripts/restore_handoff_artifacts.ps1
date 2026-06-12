param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$ResolvedZip = Resolve-Path $ZipPath
$TempPath = Join-Path $env:TEMP ("specific_range_handoff_restore_" + [guid]::NewGuid().ToString("N"))

New-Item -ItemType Directory -Path $TempPath | Out-Null
Expand-Archive -LiteralPath $ResolvedZip -DestinationPath $TempPath -Force

$Entries = Get-ChildItem -Path $TempPath -Recurse -File | Where-Object {
    $_.Name -ne "ARTIFACT_MANIFEST.txt"
}

foreach ($Entry in $Entries) {
    $Relative = [System.IO.Path]::GetRelativePath($TempPath, $Entry.FullName)
    $Destination = Join-Path $Root $Relative
    New-Item -ItemType Directory -Path (Split-Path $Destination -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $Entry.FullName -Destination $Destination -Force
    Write-Host "restored $Relative"
}

Remove-Item -LiteralPath $TempPath -Recurse -Force
Write-Host "artifact restore complete"

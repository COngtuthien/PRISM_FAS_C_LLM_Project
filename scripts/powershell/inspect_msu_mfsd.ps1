param(
    [Parameter(Mandatory = $true)][string]$RawRoot,
    [Parameter(Mandatory = $true)][string]$ReportsRoot
)
$ErrorActionPreference = 'Stop'
$raw = [System.IO.Path]::GetFullPath($RawRoot)
$out = Join-Path ([System.IO.Path]::GetFullPath($ReportsRoot)) 'raw_audit'
if (-not (Test-Path -LiteralPath $raw -PathType Container)) { throw "Raw root is not a directory: $raw" }
New-Item -ItemType Directory -Force -Path $out | Out-Null
$files = @(Get-ChildItem -LiteralPath $raw -File -Recurse -Force)
$byExtension = $files | Group-Object { if ($_.Extension) { $_.Extension.ToLowerInvariant() } else { '<none>' } } | ForEach-Object {
    [ordered]@{ extension=$_.Name; file_count=$_.Count; total_bytes=[int64](($_.Group | Measure-Object Length -Sum).Sum) }
}
$samples = $files | Sort-Object FullName | Select-Object -First 100 | ForEach-Object { $_.FullName.Substring($raw.Length).TrimStart('\') }
$metadata = $files | Where-Object { $_.Name -match '(?i)readme|protocol|train|test|dev|split|label|ground[ _-]?truth|client|attack|real|fake' -or $_.Extension -match '(?i)^\.(txt|csv|json|xml|mat)$' }
$textMetadata = foreach ($file in $metadata | Where-Object { $_.Extension -match '(?i)^\.(txt|csv|json|xml)$' -and $_.Length -le 1048576 }) {
    try { [ordered]@{ path=$file.FullName.Substring($raw.Length).TrimStart('\'); content=(Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8) } }
    catch { [ordered]@{ path=$file.FullName.Substring($raw.Length).TrimStart('\'); read_error=$_.Exception.Message } }
}
$videos = @($files | Where-Object { $_.Extension -match '(?i)^\.(avi|mov|mp4)$' } | Sort-Object FullName)
$videoMetadata = foreach ($video in ($videos | Select-Object -First 5)) {
    $entry = [ordered]@{ path=$video.FullName.Substring($raw.Length).TrimStart('\'); bytes=$video.Length }
    $entry.ffprobe_error='ffprobe is unavailable on this host; no video decoding or frame extraction was performed.'
    $entry
}
$archives = $files | Where-Object { $_.Extension -match '(?i)^\.(zip|rar|7z|001)$' } | ForEach-Object { $_.FullName.Substring($raw.Length).TrimStart('\') }
$inventory = [ordered]@{
    raw_root=$raw; generated_at=(Get-Date).ToUniversalTime().ToString('o'); top_level=@(Get-ChildItem -LiteralPath $raw -Force | ForEach-Object Name)
    file_count=$files.Count; extensions=@($byExtension); sample_paths=@($samples); named_metadata_paths=@($metadata | ForEach-Object { $_.FullName.Substring($raw.Length).TrimStart('\') }); text_metadata=@($textMetadata); video_metadata=@($videoMetadata); archive_paths=@($archives)
}
$inventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out 'msu_mfsd_file_inventory.json') -Encoding UTF8
$samples | Set-Content -LiteralPath (Join-Path $out 'msu_mfsd_sample_paths.txt') -Encoding UTF8
$lines = @('# MSU-MFSD read-only deep inspection','',"- Raw root: ``$raw``","- Files: $($files.Count)","- Top-level: $($inventory.top_level -join ', ')",'','## Extensions','')
$lines += $byExtension | ForEach-Object { "- $($_.extension): $($_.file_count) files; $($_.total_bytes) bytes" }
$lines += @('','## Named metadata candidates','') + ($inventory.named_metadata_paths | ForEach-Object { "- ``$_``" })
$lines += @('','## Video metadata (first 5; headers only)','') + ($videoMetadata | ForEach-Object { "- ``$($_.path)``: $($_.ffprobe | ConvertTo-Json -Compress) $($_.ffprobe_error)" })
$lines += @('','## Archives','') + ($archives | ForEach-Object { "- ``$_``" })
$lines | Set-Content -LiteralPath (Join-Path $out 'msu_mfsd_deep_inspection.md') -Encoding UTF8
Write-Output (Join-Path $out 'msu_mfsd_file_inventory.json')

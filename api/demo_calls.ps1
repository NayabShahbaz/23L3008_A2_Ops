# Task 7 evidence: call the deployed model API from the terminal (Figs 7.3-7.5).
# Reads API_KEY from .env and never prints it.   Run:  & .\api\demo_calls.ps1
param(
    [string]$BaseUrl = "https://api-23l3008-a2-ops.onrender.com",
    [string]$Image = "data/raw/AiArtData/images89.jpg"
)

$key = (Select-String -Path .env -Pattern '^API_KEY=(.*)$').Matches[0].Groups[1].Value.Trim()
$masked = $key.Substring(0, [Math]::Min(2, $key.Length)) + "****"

function Show($title, $cmd, $out) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
    Write-Host "> $cmd" -ForegroundColor DarkGray
    $lines = $out -split "`n"
    $status = $lines[-1]
    $body = ($lines[0..($lines.Count - 2)] -join "`n")
    try { $body = $body | ConvertFrom-Json | ConvertTo-Json } catch {}
    Write-Host $body
    $color = if ($status -like "*200*") { "Green" } else { "Yellow" }
    Write-Host $status -ForegroundColor $color
}

Write-Host "VisionGuard model API - $BaseUrl   ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White

$out = curl.exe -s -m 120 -w "`nHTTP %{http_code}  (%{time_total}s)" "$BaseUrl/health"
Show "7.3  GET /health (no key needed)" "curl $BaseUrl/health" ($out -join "`n")

$out = curl.exe -s -m 120 -w "`nHTTP %{http_code}  (%{time_total}s)" -H "X-API-Key: $key" -F "image=@$Image" "$BaseUrl/predict"
Show "7.4  POST /predict WITH X-API-Key" "curl -H 'X-API-Key: $masked' -F image=@$Image $BaseUrl/predict" ($out -join "`n")

$out = curl.exe -s -m 120 -w "`nHTTP %{http_code}  (%{time_total}s)" -F "image=@$Image" "$BaseUrl/predict"
Show "7.5  POST /predict WITHOUT key" "curl -F image=@$Image $BaseUrl/predict" ($out -join "`n")
Write-Host ""

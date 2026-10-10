param(
    [string]$InstallDir = (Join-Path $PSScriptRoot "..\external\legal-tools")
)

$ErrorActionPreference = "Stop"
$resolvedTarget = [System.IO.Path]::GetFullPath($InstallDir)
$venvPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"))
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "py" }
$serverFiles = Get-ChildItem -LiteralPath $resolvedTarget -Directory | ForEach-Object {
    $candidate = Join-Path $_.FullName "scripts\server.py"
    if (Test-Path -LiteralPath $candidate) { Get-Item -LiteralPath $candidate }
}
$flkServer = $serverFiles | Where-Object {
    Select-String -LiteralPath $_.FullName -Pattern 'flk_npc_mcp' -Quiet
} | Select-Object -First 1
$caseServer = $serverFiles | Where-Object {
    Select-String -LiteralPath $_.FullName -Pattern 'rmfyalk_mcp' -Quiet
} | Select-Object -First 1
if (-not $flkServer -or -not $caseServer) {
    throw "Legal MCP servers are not installed. Run .\scripts\setup_legal_mcps.ps1 first."
}

$existingListeners = @(Get-NetTCPConnection -State Listen -LocalPort 18061,18062 -ErrorAction SilentlyContinue)
if ($existingListeners.Count -eq 2) {
    Write-Host "Legal MCP servers are already listening on ports 18061 and 18062."
    return
}
if ($existingListeners.Count -gt 0) {
    throw "Only one legal MCP port is available. Stop the existing process and start both servers together."
}

$flkProcess = Start-Process $pythonExe -ArgumentList @("`"$($flkServer.FullName)`"") `
    -WorkingDirectory $flkServer.DirectoryName -WindowStyle Hidden -PassThru
$caseProcess = Start-Process $pythonExe -ArgumentList @("`"$($caseServer.FullName)`"") `
    -WorkingDirectory $caseServer.DirectoryName -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 1
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort 18061,18062 -ErrorAction SilentlyContinue)
if ($listeners.Count -ne 2) {
    throw "One or more legal MCP servers exited during startup."
}

$flkPid = ($listeners | Where-Object LocalPort -eq 18062).OwningProcess
$casePid = ($listeners | Where-Object LocalPort -eq 18061).OwningProcess
Write-Host "flk-npc MCP started: PID=$flkPid, http://127.0.0.1:18062/mcp"
Write-Host "rmfyalk MCP started: PID=$casePid, http://127.0.0.1:18061/mcp"
Write-Host "Stop with: Stop-Process -Id $flkPid,$casePid"

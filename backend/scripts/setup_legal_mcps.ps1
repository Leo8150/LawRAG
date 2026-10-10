param(
    [string]$InstallDir = (Join-Path $PSScriptRoot "..\external\legal-tools")
)

$ErrorActionPreference = "Stop"
$resolvedParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $InstallDir))
$resolvedTarget = [System.IO.Path]::GetFullPath($InstallDir)
$venvPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"))
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "py" }
if (-not $resolvedTarget.StartsWith($resolvedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Invalid MCP installation target: $resolvedTarget"
}

if (-not (Test-Path -LiteralPath $resolvedTarget)) {
    New-Item -ItemType Directory -Force -Path $resolvedParent | Out-Null
    git clone --depth 1 https://github.com/moyupeng0422/legal-tools.git $resolvedTarget
}

$serverDirs = Get-ChildItem -LiteralPath $resolvedTarget -Directory | Where-Object {
    Test-Path -LiteralPath (Join-Path $_.FullName "scripts\server.py")
}
$flkDir = $serverDirs | Where-Object {
    Select-String -LiteralPath (Join-Path $_.FullName "scripts\server.py") -Pattern 'flk_npc_mcp' -Quiet
} | Select-Object -First 1
$caseDir = $serverDirs | Where-Object {
    Select-String -LiteralPath (Join-Path $_.FullName "scripts\server.py") -Pattern 'rmfyalk_mcp' -Quiet
} | Select-Object -First 1
if (-not $flkDir -or -not $caseDir) {
    throw "Could not discover the two legal MCP server directories."
}

$flkRequirements = Join-Path $flkDir.FullName "requirements.txt"
$caseRequirements = Join-Path $caseDir.FullName "requirements.txt"
& $pythonExe -m pip install -r $flkRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install flk-npc MCP dependencies." }
& $pythonExe -m pip install -r $caseRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install rmfyalk MCP dependencies." }

Write-Host "Legal MCP dependencies installed at $resolvedTarget"
Write-Host "Start them with: .\scripts\start_legal_mcps.ps1"

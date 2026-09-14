[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

Push-Location (Join-Path $root "backend")
try {
    Invoke-Checked { uv sync --all-groups } "Backend dependency sync"
    Invoke-Checked { uv run ruff check . } "Backend lint"
    Invoke-Checked { uv run ruff format --check . } "Backend format check"
    Invoke-Checked { uv run mypy src } "Backend type check"
    Invoke-Checked { uv run pytest } "Backend tests"
} finally {
    Pop-Location
}

Push-Location (Join-Path $root "frontend")
try {
    Invoke-Checked { npm ci } "Frontend dependency install"
    Invoke-Checked { npm run lint } "Frontend lint"
    Invoke-Checked { npm run typecheck } "Frontend type check"
    Invoke-Checked { npm run test:run } "Frontend tests"
    Invoke-Checked { npm run build } "Frontend build"
} finally {
    Pop-Location
}

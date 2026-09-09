[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location (Join-Path $root "backend")
try {
    uv sync --all-groups
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    uv run pytest
} finally {
    Pop-Location
}

Push-Location (Join-Path $root "frontend")
try {
    npm ci
    npm run lint
    npm run typecheck
    npm run test -- --run
    npm run build
} finally {
    Pop-Location
}

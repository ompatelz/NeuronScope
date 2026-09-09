# NeuronScope

NeuronScope is an interactive debugger for small neural-network training runs. It will make the internals of a model visible: decision boundaries, activations, gradients, weights, losses, and transparent diagnostics.

The project is intentionally built in small, explainable steps. The current bootstrap provides a FastAPI service boundary and a React workbench shell; it does not yet train models or display invented telemetry.

## Repository layout

```text
backend/       FastAPI transport layer and Python test suite
frontend/      React + TypeScript workbench
docs/          Architecture decisions and learning notes
.github/       Continuous-integration workflows
```

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 24 or newer and npm

## Run locally

Start the API:

```powershell
Set-Location backend
uv sync --all-groups
uv run uvicorn neuronscope.main:app --reload
```

In another terminal, start the workbench:

```powershell
Set-Location frontend
npm install
npm run dev
```

The API health check is available at `http://127.0.0.1:8000/api/v1/health`.

## Verification

From the repository root, run the complete local gate:

```powershell
.\scripts\verify.ps1
```

Or run an individual application’s checks:

```powershell
Set-Location backend
uv run ruff check .
uv run mypy src
uv run pytest

Set-Location ../frontend
npm run lint
npm run typecheck
npm run test -- --run
npm run build
```

## Roadmap

The next tasks add deterministic 2D dataset generation, a configurable PyTorch MLP, and an instrumented training loop before a data-rich workbench is built.

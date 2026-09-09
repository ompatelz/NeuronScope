# Architecture

NeuronScope starts as a modular monolith: one browser client and one Python API in the same repository. This keeps local development simple while preserving the boundaries the debugger needs.

```text
React workbench  ->  FastAPI API  ->  dataset/model/training modules
                                         -> instrumentation -> diagnostics
```

## Current boundary

- `frontend/` owns rendering, interactions, and typed API clients.
- `backend/src/neuronscope/api/` owns HTTP routing and request/response schemas.
- Future `datasets`, `models`, `training`, `instrumentation`, and `diagnostics` packages own domain logic. API routes must orchestrate them rather than contain their implementation.

The health endpoint is deliberately the only runtime behavior in the bootstrap. It verifies that the transport boundary works without implying a training implementation that does not exist.

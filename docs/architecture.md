# Architecture

NeuronScope starts as a modular monolith: one browser client and one Python API in the same repository. This keeps local development simple while preserving the boundaries the debugger needs.

```text
React workbench  ->  FastAPI API  ->  dataset/model/training modules
                                         -> instrumentation -> diagnostics
```

## Current boundary

- `frontend/` owns rendering, interactions, and typed API clients.
- `backend/src/neuronscope/api/` owns HTTP routing and request/response schemas.
- `datasets`, `models`, and `training` own deterministic data generation, model construction, and
  synchronous optimization respectively. Future `instrumentation` and `diagnostics` packages will
  observe those domains without folding their logic into the trainer. API routes must orchestrate
  domain services rather than contain their implementation.

The training engine consumes typed dataset results and a configured MLP directly. It stays
transport-independent: the later experiment API can call it, while tests or notebooks can reuse
the same service without constructing HTTP requests.

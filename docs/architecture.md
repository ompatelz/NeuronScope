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

## Experiment request flow

`POST /api/v1/experiments` accepts nested dataset, model, and training configurations. Pydantic
enforces their resource and shape bounds before an application service generates the dataset,
builds the MLP, and runs training synchronously. The response contains the generated points,
serializable architecture metadata, per-epoch metrics, optional scalar instrumentation, and final
quality values. The route owns only HTTP transport; orchestration remains reusable without HTTP.

Runs intentionally stay synchronous and CPU-bound while datasets, layer widths, layer counts, and
epoch counts remain small and validated. NeuronScope does not need a queue or distributed worker
for this product boundary.

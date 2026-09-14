# Contributing to NeuronScope

Thanks for improving NeuronScope. Keep changes focused on inspectable, bounded neural-network
experiments. Visualizations must be backed by real API data; do not add fabricated telemetry or
paths that execute uploaded model code.

## Development workflow

1. Create a focused branch from the latest `main`.
2. Add or update tests with the implementation.
3. Run `./scripts/verify.ps1` on Windows, or the backend and frontend commands documented in the
   README on other platforms.
4. Update architecture, deployment, or learning documentation when behavior changes.
5. Open a pull request using the repository checklist and explain any unverified integration edge.

Keep commits small and descriptive. Never commit credentials, local environment files, generated
build output, model binaries, or user data.

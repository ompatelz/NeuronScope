# Deployment

NeuronScope ships as two containers: a single-worker FastAPI backend and an unprivileged Nginx
frontend. Only Nginx is published to the host. Requests under `/api/` are proxied to the private
backend service, so the browser uses the same origin and no production CORS configuration is
required.

## Local production build

Prerequisites: Docker Engine with Compose v2 and BuildKit.

```powershell
docker compose build
docker compose up --detach
docker compose ps
```

Open `http://127.0.0.1:8080`. To use another host port:

```powershell
$env:NEURONSCOPE_PORT = "8090"
docker compose up --detach
```

Check the public frontend and proxied backend health endpoints:

```powershell
Invoke-WebRequest http://127.0.0.1:8080/healthz
Invoke-RestMethod http://127.0.0.1:8080/api/v1/health
```

Inspect logs and stop the deployment:

```powershell
docker compose logs --follow
docker compose down
```

No `.env` file, credential, or API key is required. Compose passes only the non-sensitive
application display name. Both runtime images use read-only filesystems, writable temporary
mounts, health checks, and non-root application processes.

## Hosting guidance

Deploy `compose.yaml` on a container host, or deploy the two Dockerfiles as services on a platform
that supports a private service network. Route public traffic only to frontend port 8080. Preserve
the service name `backend`, or replace the Nginx upstream with the platform's private backend DNS
name.

Terminate TLS at the hosting platform or an external reverse proxy. Configure request-rate and
concurrency limits before exposing the synchronous experiment endpoint publicly. NeuronScope
accepts only its bounded dataset/model/training schemas, but training is CPU-intensive and the
portfolio deployment is not intended as a multi-tenant compute service.

The backend intentionally runs one Uvicorn worker. Multiple workers multiply PyTorch memory usage;
scale only after measuring the host's CPU and memory limits. Do not add arbitrary uploaded-model
execution to the public deployment.

## Release verification

Before publishing an image, run the repository quality gate, build without stale layers, and verify
both health endpoints through Nginx:

```powershell
.\scripts\verify.ps1
docker compose build --pull --no-cache
docker compose up --detach --wait
docker compose ps
Invoke-RestMethod http://127.0.0.1:8080/api/v1/health
```

Pin deployed image digests in the hosting environment for reproducible releases. The repository
pins runtime major/minor families while retaining security patch updates during fresh builds.

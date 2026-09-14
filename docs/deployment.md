# Deployment

NeuronScope's local production topology uses two containers: a single-worker FastAPI backend and
an unprivileged Nginx frontend. Only Nginx is published to the host. Requests under `/api/` are
proxied to the private backend service, so the browser uses the same origin and no production CORS
configuration is required. The recommended public topology below packages the same application as
one service.

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

## Recommended public deployment: one Railway service

For the first public release, use one Railway Hobby service built from the repository's root
production Dockerfile. That image bundles the compiled React client with the FastAPI/PyTorch API,
so the browser and `/api/` share one origin. This is simpler and cheaper to operate than exposing
the existing Compose frontend and backend as two separately billed services. Keep `compose.yaml`
for local production checks.

This recommendation is for the CPU-only, bounded teaching workload in this repository. It is not
an endorsement for arbitrary model training or a multi-tenant inference service.

### Cost and initial sizing

The following figures are planning estimates checked on **September 14, 2026**, not a quote:

- Railway Hobby has a **$5 USD monthly floor**, with that $5 applied to resource usage. Railway
  currently lists usage rates of $10 per GB-month of memory and $20 per vCPU-month of CPU. Review
  the current [Railway plans and resource pricing](https://docs.railway.com/pricing/plans) before
  enabling billing.
- Start with a ceiling of **1 vCPU, 2 GB RAM, one replica, and one Uvicorn worker**. The 2 GB figure
  is an engineering estimate for the CPU builds of PyTorch, NumPy, and scikit-learn plus one bounded
  training request; it is not a measured production minimum. Inspect peak memory and request
  duration after real traffic, then lower or raise the limits deliberately.
- Do not start on the 512 MB Free allocation. It leaves too little safety margin for the scientific
  Python runtime and a training request, even if the container happens to boot.
- Set Railway usage alerts and a hard usage limit before sharing the URL. These controls and their
  project-level behavior are described in Railway's [cost-control guide](https://docs.railway.com/pricing/cost-control).

The service-level CPU and memory settings are hard ceilings, not admission control. NeuronScope's
training endpoint is synchronous and CPU-intensive. Keep one replica and worker initially, avoid
overlapping public training requests, and retain the built-in workload budget and one-job admission
gate. Add edge rate limiting before broad or adversarial traffic; the Compose Nginx configuration
shows a conservative starting policy. Multiple Uvicorn workers multiply PyTorch memory use. Never
expose arbitrary uploaded-model or code execution.

### Create the service in the Railway dashboard

Publishing changes external state and can incur charges. Before proceeding, the repository owner
must explicitly authorize both:

1. the exact source and destination—this GitHub repository and branch into a named Railway
   project/environment; and
2. activation of the Railway Hobby subscription and its usage billing.

After that authorization:

1. In Railway, create a project and choose **Deploy from GitHub repo**. Grant the Railway GitHub App
   access only to the intended private repository, select the intended production branch, and create
   one service. Railway supports GitHub repositories as a service source and deploys new commits;
   see [Services](https://docs.railway.com/services).
2. Leave the service root at the repository root so Railway finds the root production Dockerfile.
   Do not point this single service at `backend/Dockerfile` or `frontend/Dockerfile`; those images are
   the separate Compose deployment units.
3. Do not add `railway.json`, `railway.toml`, or project Infrastructure-as-Code for this initial
   deployment. Set the build, resource, health, networking, and serverless options explicitly in the
   dashboard so the owner can review them before they take effect.
4. Under the service's resource settings, start with one replica, 1 vCPU, and 2 GB RAM. Enable
   **Wait for CI** if available so a failing GitHub workflow does not race a production deploy.
5. Configure the health-check path as `/api/v1/health`. Railway injects `PORT`; the container must
   listen on `0.0.0.0:$PORT`, and Railway checks that same port. A successful deployment health
   check gates traffic switching, but it is not continuous uptime monitoring. See Railway's
   [health-check behavior](https://docs.railway.com/deployments/healthchecks).
6. In **Networking → Public Networking**, generate a temporary `*.up.railway.app` domain. Railway
   terminates HTTPS automatically. For a custom domain, add both DNS records Railway displays and
   wait for certificate/domain verification; see [Public Networking](https://docs.railway.com/networking/public-networking)
   and [custom-domain setup](https://docs.railway.com/networking/domains/working-with-domains).

No application secret or database is required for the current stateless release. Treat the
container filesystem as ephemeral and do not rely on it for experiment persistence.

### Serverless mode and cold starts

For a low-traffic portfolio launch, enable Railway Serverless on the service. Railway currently
documents that an inactive service sleeps after 10 minutes and wakes on the next request; see
[Cut Idle Costs with Serverless Mode](https://docs.railway.com/guides/cut-idle-costs-serverless).
Sleeping reduces idle compute usage, but the first visitor pays the container and PyTorch cold-start
latency. Disable Serverless if consistently fast first responses matter more than idle cost. In
either mode, the Hobby subscription floor still applies.

### Verify and operate the release

After the first deployment:

1. Open the generated HTTPS URL and complete one healthy-baseline training run.
2. Request `https://<domain>/api/v1/health` and require a `2xx` response.
3. Inspect both build and runtime logs for dependency, bind-address, out-of-memory, timeout, and
   restart errors. Watch CPU, memory, response duration, and concurrent-request behavior during the
   training run.
4. Confirm a client-side route refresh and the `/api/` same-origin proxy both work through HTTPS.
5. If a release fails, use the service's **Deployments** history to view its logs. Roll back to the
   last known-good deployment when it is still retained; Railway restores the earlier image and
   custom variables. Otherwise, redeploy that source revision. See [Deployment Actions](https://docs.railway.com/deployments/deployment-actions)
   and the [rollback guide](https://docs.railway.com/guides/roll-back-bad-deploy).

Revisit the 1 vCPU/2 GB ceiling using observed data rather than increasing replicas automatically.
If concurrent public use becomes a real requirement, move training to a bounded queue/worker design
before horizontal scaling.

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

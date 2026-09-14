# syntax=docker/dockerfile:1.7

FROM node:24-alpine3.22 AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY index.html tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts ./
COPY src ./src
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.28-alpine3.21 AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder --chown=101:101 /app/dist /usr/share/nginx/html

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=4 \
    CMD ["wget", "--quiet", "--output-document=/dev/null", "http://127.0.0.1:8080/healthz"]

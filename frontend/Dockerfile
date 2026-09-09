# syntax=docker/dockerfile:1.7
#
# Frontend image for the Next.js 16 chat + pipeline-inspector UI.
# Built from the frontend/ directory as build-context.
#
#   docker build -t bilingual-crag-frontend .
#
# NEXT_PUBLIC_API_BASE is baked in at BUILD time (not runtime) because Next.js
# inlines NEXT_PUBLIC_* vars into the client-side bundle during `next build`.
# Default points at the host's localhost:8000 (where the backend container's
# port is exposed); override via --build-arg for production deploys.

ARG NEXT_PUBLIC_API_BASE=http://localhost:8000

# Pin pnpm to the version that wrote pnpm-lock.yaml locally (11.8.0).
# `corepack prepare pnpm@latest` can pull a newer version whose lockfile
# format diverges in subtle ways -- then --frozen-lockfile fails the build.
ARG PNPM_VERSION=11.8.0

# ---- Stage 1: install deps (cached layer) -----------------------------------
FROM node:22-alpine AS deps
ARG PNPM_VERSION
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@${PNPM_VERSION} --activate
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

# ---- Stage 2: build (produces .next/standalone) -----------------------------
FROM node:22-alpine AS builder
ARG PNPM_VERSION
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@${PNPM_VERSION} --activate
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

# ---- Stage 3: minimal runtime ------------------------------------------------
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Non-root user for the runtime stage.
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

CMD ["node", "server.js"]

# ============================================================
#  agentic-research-analyst  ·  Production Container
# ============================================================
#  Multi-stage build: slim runtime image, non-root user,
#  health-check endpoint, ~380 MB final image.
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="RFNOACH"
LABEL org.opencontainers.image.source="https://github.com/RFNOACH/agentic-research-analyst"
LABEL org.opencontainers.image.description="Multi-tool LLM research agent"

# Security: non-root user
RUN groupadd --gid 1000 agent \
    && useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY --chown=agent:agent . .

# Sandbox directory for code execution tool
RUN mkdir -p /app/sandbox && chown agent:agent /app/sandbox

USER agent

# Expose FastAPI port
EXPOSE 8000

# Health check against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

# Default: run the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# open-deepthink — production-oriented image
# Build:  docker build -t open-deepthink .
# Run:    docker run --rm -p 8000:8000 -e OPENROUTER_API_KEY=... open-deepthink

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# System deps for faiss / scientific stack wheels (prebuilt usually enough)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install package first (better layer caching)
COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY deepthink ./deepthink
COPY app.py index.html launch.bat ./
COPY css ./css
COPY js ./js
COPY static ./static
COPY skills ./skills

RUN pip install --upgrade pip \
    && pip install -e .

# Runtime state directories (also volume-mounted in compose)
RUN mkdir -p /app/distillation_output /app/.deepthink-state /app/skills

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null || exit 1

# Prefer console script; falls back to module
CMD ["python", "-m", "deepthink"]

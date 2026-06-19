FROM python:3.12-slim

# Unbuffered stdout so print()/logs appear in `docker logs` immediately.
ENV PYTHONUNBUFFERED=1

# Tools the agent's `bash` may reasonably reach for when analyzing data.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user whose uid/gid match the host (default 1000) so files
# written to bind-mounted volumes (data/, workspace/) are owned by you, not root.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app

WORKDIR /app

# Install Python deps first for layer caching. Memory runs as a separate mem0
# service (talked to over HTTP), so this image needs no torch/chromadb.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "openai>=1.30" \
        "httpx>=0.27" "python-dotenv>=1.0" "pyyaml>=6.0" "mcp>=1.0" \
        "python-multipart>=0.0.9" "discord.py>=2.3,<3" "croniter>=2.0"

USER app

# App code. Runtime data (.env, soul.md, skills, data, workspace) is bind-mounted.
COPY --chown=app:app app ./app
COPY --chown=app:app web ./web

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# mem0ai + pgvector client. No torch: embeddings come from the fastembed service
# over HTTP (embedder provider "openai"), so nothing heavy is installed here.
RUN pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" \
        "mem0ai>=0.1.48" "psycopg[binary,pool]>=3.2.8"

RUN mkdir -p /app/history
COPY server.py ./

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

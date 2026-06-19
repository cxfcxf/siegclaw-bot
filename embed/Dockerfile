FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# fastembed pulls onnxruntime (lightweight — no torch/CUDA), tokenizers, numpy.
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27" "fastembed>=0.3"

# Bake the embedding model into the image so the first request isn't slow and the
# container needs no network at runtime.
ARG EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}')"

COPY server.py ./

EXPOSE 8090
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY eval ./eval
COPY data ./data
RUN pip install --no-cache-dir -e .

# Bake the embedding model into the image at build time (network access here
# is normal for a build step) so the container never needs network access at
# runtime and doesn't pay a multi-second cold-start on its first request -
# api/server.py already warms the retriever at startup; this just makes sure
# the model is already sitting in the image when that runs.
RUN python -c "from copilot.rag.embeddings import get_embedder; get_embedder()"

ENV COPILOT_AUDIT_DB=/app/data/audit.db
ENV COPILOT_CHECKPOINT_DB=/app/data/checkpoints.db
ENV COPILOT_TRACE_LOG=/app/data/traces.jsonl

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

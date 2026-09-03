FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY eval ./eval
COPY data ./data
RUN pip install --no-cache-dir -e .

ENV COPILOT_AUDIT_DB=/app/data/audit.db
ENV COPILOT_CHECKPOINT_DB=/app/data/checkpoints.db

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

LABEL maintainer="Equipo 56 MLOps"
LABEL version="1.0.0"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app_api.py .
COPY mlops/ ./mlops/

# DVC metadata (to allow runtime dvc pull)
COPY dvc.yaml dvc.lock ./
COPY .dvc/ .dvc/

# Entrypoint script to setup AWS profile and pull artifacts via DVC at runtime
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

RUN mkdir -p mlruns reports/drift

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]

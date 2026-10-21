FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SCENE_PUBLIC=1 HF_HUB_DISABLE_TELEMETRY=1 ANONYMIZED_TELEMETRY=False \
    EMBEDDING_CACHE_DIR=/opt/embedding-cache PORT=10000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 app \
    && mkdir -p /opt/embedding-cache /app/data \
    && chown -R app:app /app /opt/embedding-cache
COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt
COPY --chown=app:app . .
USER app
RUN python deployment.py prepare
USER root
RUN mv /app/data /opt/seed-data && mkdir /app/data && chown app:app /app/data
USER app
EXPOSE 10000
CMD ["python", "deployment.py", "serve"]

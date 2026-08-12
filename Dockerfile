# Container image for the fraud detection API.
#
# Only what the API needs goes in: the serving dependencies, the app package
# and the trained pipeline. The DVC/MLflow/Evidently stack is part of the
# training and monitoring workflow, not of serving, so it stays out and the
# image stays small enough to build and boot on Render's free tier.

FROM python:3.11-slim

# Faster, quieter, and no stale .pyc files in the layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/srv/app/models/model.pkl \
    PORT=8000

WORKDIR /srv/app

# Dependencies first so this layer is cached while application code changes.
# requirements-api.txt is fully pinned, so a rebuild installs the same
# versions the model was trained and tested against.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code and the trained Phase 1 pipeline.
COPY app/ ./app/
COPY models/model.pkl ./models/model.pkl

# Run as a non-root user.
RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /srv/app
USER apiuser

EXPOSE 8000

# No secrets are baked into this image: it contains code and a model file only.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:'+os.getenv('PORT','8000')+'/health'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"

# Render injects $PORT; the shell form lets us honour it, with 8000 locally.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

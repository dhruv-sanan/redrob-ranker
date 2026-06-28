FROM python:3.11-slim AS sandbox

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-app.txt ./
RUN pip install -r requirements.txt -r requirements-app.txt

COPY src/ ./src/
COPY tools/ ./tools/
COPY config/ ./config/
COPY app.py rank.py build_features.py reasoning_audit.py ./

# Vendor the BGE-small model at build time so cold-start is fast.
RUN python tools/vendor_model.py --out-dir artifacts/model

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/').read()" || exit 1

CMD ["python", "app.py"]

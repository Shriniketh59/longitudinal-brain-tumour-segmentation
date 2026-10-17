# NeuroScan AI backend: FastAPI + PyTorch (CPU) + MONAI.
# Built for Hugging Face Spaces (Docker SDK, port 7860, runs as uid 1000); works on any Docker host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch wheels (the default CUDA wheels add ~2 GB and are useless without a GPU).
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && grep -vE '^(torch|torchvision|pytest)' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY --chown=user app ./app
COPY --chown=user scripts/__init__.py scripts/download_brats.py ./scripts/
COPY --chown=user weights ./weights

USER user
ENV HOME=/home/user \
    CORS_ORIGINS="http://localhost:3000"

# Sample BraTS 2023 cases (CC BY 4.0) for the 3D lab and longitudinal demos.
ARG BRATS_CASES=10
RUN python -m scripts.download_brats --cases ${BRATS_CASES}

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

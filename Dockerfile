# CPU-only image for reproducible benchmarks. No GPU, no CUDA.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
RUN python scripts/download_assets.py \
    && for m in yolox_nano yolox_tiny yolox_s; do python scripts/quantize.py models/$m.onnx; done

# Headless by default: a container has no display or webcam.
CMD ["python", "benchmark.py"]

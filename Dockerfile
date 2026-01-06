FROM python:3.11-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Install CPU-only PyTorch FIRST (better layer caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir \
    transformers>=4.55.0 \
    accelerate \
    sentencepiece \
    protobuf \
    fastapi \
    uvicorn[standard] \
    requests \
    python-multipart

# Copy the local model directory (do this AFTER deps for better caching)
COPY LFM2-1.2B /models/LFM2-1.2B

# Copy application code
COPY *.py ./

# Expose port
EXPOSE 8090

# Run with proper signal handling
CMD ["python", "-u", "server.py"]
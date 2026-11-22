# # Python 3.10 slim base
# FROM python:3.10-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     DEBIAN_FRONTEND=noninteractive \
#     PORT=8000

# WORKDIR /app

# # Install system dependencies for OpenCV and video codecs
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install CPU-optimized PyTorch first (saves ~3GB image size)
# COPY requirements.txt /app/
# RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy backend code
# COPY . /app/

# # Ensure output and weight directories exist
# RUN mkdir -p uploads outputs app/weights

# EXPOSE 8000

# # Download weights from Hugging Face if not present, then boot FastAPI on dynamic $PORT
# CMD ["sh", "-c", "python download_weights.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]






# Python 3.10 slim base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV and video codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install CPU-optimized PyTorch first (saves ~3GB image size)
COPY requirements.txt /app/
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/

# Ensure output and weight directories exist
RUN mkdir -p uploads outputs app/weights

EXPOSE 8000

# Download weights from Hugging Face if not present, then boot FastAPI on dynamic $PORT
CMD ["sh", "-c", "python download_weights.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
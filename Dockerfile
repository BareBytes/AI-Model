# Use an official Python 3.10 runtime as a base image
FROM python:3.10-slim

# Set environment variables to prevent buffer issues and speed up installation
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Copy all the files from the current folder to the working directory in the container
COPY . /app

# Install git, cmake, and dependencies required for dlib
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and install wheel using the default PyPI index
RUN pip install --upgrade pip==24.2 setuptools wheel -i https://pypi.org/simple

# Install PyTorch packages from the PyTorch CPU index
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies from PyPI
RUN pip install --no-cache-dir git+https://github.com/ultralytics/ultralytics.git@main \
    opencv-python \
    opencv-python-headless \
    ultralytics \
    cmake \
    face_recognition \
    dlib

# Clone the YOLOv10 repository
RUN git clone https://github.com/THU-MIG/yolov10.git

# Set entrypoint if needed (optional)
ENTRYPOINT ["python", "Login.py"]

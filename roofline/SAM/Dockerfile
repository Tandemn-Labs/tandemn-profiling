# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    flask==3.1.2 \
    pillow==12.1.0
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git
RUN mkdir -p /app/checkpoints && \
    cd /app/checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
RUN python3 -c "import sam2; import os; import shutil; \
    sam2_path = os.path.dirname(sam2.__file__); \
    shutil.copytree(os.path.join(sam2_path, 'configs'), '/app/configs')"
COPY SAM_2_server.py /app/
EXPOSE 8080
CMD ["python3", "SAM_2_server.py", "--port", "8080", "--host", "0.0.0.0"]
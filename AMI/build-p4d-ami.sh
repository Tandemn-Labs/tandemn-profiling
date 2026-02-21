#!/usr/bin/env bash
set -euxo pipefail

# ── Step 1: System prep ────────────────────────────────────
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
apt-get install -y build-essential linux-headers-$(uname -r) dkms pkg-config

# ── Step 2: NVIDIA CUDA repo (needed for CUDA Toolkit) ────
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# ── Step 3: NVIDIA Driver 580.105.08 (.run installer) ─────
# --no-drm: kernel 6.17+ changed drm_fbdev_ttm_driver_fbdev_probe;
#           DRM-KMS is not needed for compute-only (vLLM) workloads.
wget https://download.nvidia.com/XFree86/Linux-x86_64/580.105.08/NVIDIA-Linux-x86_64-580.105.08.run
chmod +x NVIDIA-Linux-x86_64-580.105.08.run
./NVIDIA-Linux-x86_64-580.105.08.run --silent --dkms --no-drm

# ── Step 4: Fabric Manager 580.105.08 (required for p4d NVSwitch) ─
# Install exact 580.105.08 .debs from NVIDIA CUDA repo (Ubuntu apt
# repos only carry the latest patch, which would mismatch the driver).
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnvidia-nscq_580.105.08-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nvidia-fabricmanager_580.105.08-1_amd64.deb
dpkg -i libnvidia-nscq_580.105.08-1_amd64.deb
dpkg -i nvidia-fabricmanager_580.105.08-1_amd64.deb
systemctl enable --now nvidia-fabricmanager

# ── Step 5: nvidia-persistenced ────────────────────────────
# The .run installer ships the binary but not a systemd unit.
cat > /etc/systemd/system/nvidia-persistenced.service << 'PERSIST_EOF'
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-persistenced --user root
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced

[Install]
WantedBy=multi-user.target
PERSIST_EOF
systemctl daemon-reload
systemctl enable --now nvidia-persistenced

# ── Step 6: CUDA Toolkit 12.8 ─────────────────────────────
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8

# Set PATH/LD_LIBRARY_PATH system-wide
cat > /etc/profile.d/cuda.sh << 'CUDA_EOF'
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
CUDA_EOF
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# ── Step 7: Python 3.12 + vLLM environment ────────────────
apt-get install -y python3.12 python3.12-venv python3.12-dev
python3.12 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate

pip install -U pip
pip install uv

# Install PyTorch with CUDA 12.8 wheels, then xformers, then vLLM
uv pip install \
  "torch==2.7.1" \
  "torchvision==0.22.1" \
  "torchaudio==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install "xformers==0.0.31" --index-url https://download.pytorch.org/whl/cu128

uv pip install "vllm==0.10.0"

# Pin transformers to exact version (vllm pulls latest >= 4.53.2)
uv pip install "transformers==4.53.2"

# ── Step 8: Verification ──────────────────────────────────
nvidia-smi
nvidia-smi nvlink -s
nvcc --version
systemctl status nvidia-fabricmanager --no-pager
systemctl status nvidia-persistenced --no-pager
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo "AMI build complete. Stop instance and create image."

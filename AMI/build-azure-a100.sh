#!/usr/bin/env bash
set -euxo pipefail

# Azure NC A100 v4 (PCIe, no NVSwitch) — same stack as AWS minus Fabric Manager.

# ── Step 1: System prep ────────────────────────────────────
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
apt-get install -y build-essential linux-headers-$(uname -r) dkms pkg-config

# ── Step 2: NVIDIA CUDA repo ──────────────────────────────
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# ── Step 3: NVIDIA Driver 580.105.08 ─────────────────────
# --no-drm: kernel 6.14+ changed DRM symbols; not needed for compute.
# NOTE: Secure Boot must be disabled on the VM (--security-type Standard).
wget https://download.nvidia.com/XFree86/Linux-x86_64/580.105.08/NVIDIA-Linux-x86_64-580.105.08.run
chmod +x NVIDIA-Linux-x86_64-580.105.08.run
./NVIDIA-Linux-x86_64-580.105.08.run --silent --dkms --no-drm

# No Fabric Manager — NC A100 v4 is PCIe, not NVSwitch.

# ── Step 4: nvidia-persistenced ───────────────────────────
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

# ── Step 5: CUDA Toolkit 12.8 ────────────────────────────
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8

cat > /etc/profile.d/cuda.sh << 'CUDA_EOF'
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
CUDA_EOF
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# ── Step 6: Python 3.12 + vLLM environment ───────────────
apt-get install -y python3.12 python3.12-venv python3.12-dev
python3.12 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate

pip install -U pip
pip install uv

uv pip install \
  "torch==2.7.1" \
  "torchvision==0.22.1" \
  "torchaudio==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install "xformers==0.0.31" --index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm==0.10.0"
uv pip install "transformers==4.53.2"

# ── Step 7: Verification ─────────────────────────────────
nvidia-smi
nvcc --version
systemctl status nvidia-persistenced --no-pager
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

echo "Azure A100 build complete."

#!/usr/bin/env bash
# Setup vLLM 0.10.0 on AWS DLAMI (Deep Learning Base OSS Nvidia Driver GPU AMI)
# Tested on: p4d.24xlarge (8x A100 SXM4 40GB), p4de.24xlarge (8x A100 SXM4 80GB)
#
# DLAMI: "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260123"
#   AMI IDs:
#     us-east-1: ami-01f3b8bbe6f7238ca
#     us-east-2: ami-041b72a99c9d48c31
#     us-west-2: ami-04a7149ad60555d54
#
# DLAMI includes: driver 580.126.09, CUDA 12.9, EFA 1.45.0, OFI NCCL 1.17.2
#
# Usage: bash setup-dlami-a100.sh
set -euxo pipefail

echo "=== 1. GPU check ==="
nvidia-smi

echo "=== 2. Fabric Manager (required for NVSwitch on p4d/p4de) ==="
if systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
  echo "Fabric Manager already running"
else
  DRIVER_VERSION=$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader)
  DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
  echo "Driver: $DRIVER_VERSION, installing Fabric Manager..."
  sudo apt-get update -qq
  sudo apt-get install -y "nvidia-fabricmanager-${DRIVER_MAJOR}=${DRIVER_VERSION}-1" 2>/dev/null \
    || sudo apt-get install -y "nvidia-fabricmanager-${DRIVER_MAJOR}" 2>/dev/null \
    || { echo "ERROR: Could not install Fabric Manager"; exit 1; }
  sudo systemctl enable --now nvidia-fabricmanager
  sleep 3
  systemctl is-active --quiet nvidia-fabricmanager && echo "Fabric Manager: OK" \
    || { echo "ERROR: Fabric Manager failed to start"; exit 1; }
fi

echo "=== 3. NVLink topology ==="
nvidia-smi topo -m 2>&1 | head -15

echo "=== 4. EFA check ==="
fi_info -p efa -t FI_EP_RDM 2>/dev/null | head -5 || echo "EFA: not available (OK for single-node)"

echo "=== 5. Install uv ==="
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "=== 6. Create venv + install vLLM ==="
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install "vllm==0.10.0" "ray[cgraph]"
uv pip install "transformers==4.57.3"

echo "=== 7. Verify ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
python3 -c "import ray; print(f'Ray {ray.__version__}')"
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Example:  python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000"

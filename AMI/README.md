# vLLM 0.10.0 on A100 — Setup Guide

## Pinned Stack

| Component | Version |
|---|---|
| NVIDIA Driver | 580.105.08 |
| CUDA Toolkit | 12.8 |
| Python | 3.12 |
| vLLM | 0.10.0 |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| xformers | 0.0.31 |
| ray | 2.54.0 |
| transformers | 4.53.2 |

## Cloud-specific guides

- **AWS** (p4d.24xlarge, 8x A100 SXM4 40GB): see [AWS.md](AWS.md)
- **Azure** (NC96ads_A100_v4, 4x A100 PCIe 80GB): see [AZURE.md](AZURE.md)
- **GCP** (a2-ultragpu/highgpu, up to 8x A100): see [GCP.md](GCP.md)

## SkyPilot (fastest way)

```bash
# AWS (uses pre-built AMI — instant)
sky launch AMI/skypilot-aws-a100.yaml --yes --env MODEL=<model> --env TP=4

# Azure
sky launch AMI/skypilot-azure-a100.yaml --yes --env MODEL=<model> --env TP=2 --env PP=2

# GCP
sky launch AMI/skypilot-gcp-a100.yaml --yes --env MODEL=<model> --env TP=4
```

## Files

| File | Description |
|---|---|
| `build-p4d-ami.sh` | Install script for AWS p4d (includes Fabric Manager for NVSwitch) |
| `build-azure-a100.sh` | Install script for Azure NC A100 v4 (no Fabric Manager, PCIe) |
| `skypilot-aws-a100.yaml` | SkyPilot config for AWS (uses pre-built AMI) |
| `skypilot-azure-a100.yaml` | SkyPilot config for Azure |
| `skypilot-gcp-a100.yaml` | SkyPilot config for GCP |
| `AWS.md` | AWS launch/usage/rebuild instructions |
| `AZURE.md` | Azure launch/install instructions |
| `GCP.md` | GCP launch/quota instructions |

## Gotchas

1. **Driver needs `--no-drm`** on kernel 6.14+/6.17+. DRM module fails due to renamed kernel symbol. Not needed for compute.
2. **Fabric Manager must exactly match driver version.** Ubuntu apt only has latest patch (580.126.x). Download 580.105.08 `.deb` directly from NVIDIA CUDA repo.
3. **nvidia-persistenced has no systemd unit** from `.run` installer. Build scripts create one manually.
4. **`pip install vllm` pulls latest transformers** (5.x). Must pin after: `uv pip install "transformers==4.53.2"`.
5. **Install PyTorch BEFORE vllm** with `--index-url https://download.pytorch.org/whl/cu128`. Otherwise vllm pulls cu121 wheels.
6. **Azure: Secure Boot must be disabled** (`--security-type Standard` at VM creation). Cannot change after.
7. **AWS: Set `DeleteOnTermination=false`** on EBS when building AMIs.

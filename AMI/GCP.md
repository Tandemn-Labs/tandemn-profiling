# GCP — A100 VMs (a2-highgpu / a2-ultragpu)

No pre-built image. SkyPilot handles driver/CUDA, we install the Python stack.

## Prerequisites

```bash
# 1. Install GCP SDK deps (in SkyPilot venv)
pip install google-api-python-client google-cloud-storage

# 2. Enable required GCP APIs
gcloud services enable cloudresourcemanager.googleapis.com compute.googleapis.com storage.googleapis.com

# 3. Auth
gcloud auth login
gcloud auth application-default login

# 4. Restart SkyPilot API server to pick up creds
sky api stop && sky api start

# 5. Verify
sky check gcp
```

## Launch with SkyPilot

```bash
# TP=4 with Qwen3-0.6B (spot — see quota section below)
sky launch AMI/skypilot-gcp-a100.yaml --yes --env MODEL=Qwen/Qwen3-0.6B --env TP=4

# TP=2 PP=2
sky launch AMI/skypilot-gcp-a100.yaml --yes --env MODEL=Qwen/Qwen2.5-72B-Instruct --env TP=2 --env PP=2

# Tear down
sky down <cluster-name> --yes
```

## Quota (current)

**On-demand A100 quota is 1 GPU per region — not enough for any multi-GPU VM.**
**Must use spot/preemptible instances (16 GPU quota per region).**

| Quota | Limit | Enough for |
|---|---|---|
| NVIDIA_A100_GPUS (on-demand) | **1** per region | Nothing useful |
| PREEMPTIBLE_NVIDIA_A100_GPUS (spot) | **16** per region | Up to 2x a2-highgpu-8g |
| NVIDIA_L4_GPUS | 8 per region | 2x g2-standard-48 |
| NVIDIA_T4_GPUS | 4 per region | 1x n1-standard-32 + 4xT4 |

**Important**: GPU quota alone isn't enough. The VM family also needs vCPU quota:
- `a2-highgpu` (A100 40GB) — had quota, **worked with spot**
- `a2-ultragpu` (A100 80GB) — no vCPU quota, **failed even with GPU quota**

To request more quota: GCP Console > IAM & Admin > Quotas > filter by GPU type and region.

## GCP A100 VM Types

| Machine Type | GPUs | VRAM | vCPUs |
|---|---|---|---|
| a2-highgpu-1g | 1x A100 40GB | 40GB | 12 |
| a2-highgpu-2g | 2x A100 40GB | 80GB | 24 |
| a2-highgpu-4g | 4x A100 40GB | 160GB | 48 |
| a2-highgpu-8g | 8x A100 40GB | 320GB | 96 |
| a2-ultragpu-1g | 1x A100 80GB | 80GB | 12 |
| a2-ultragpu-2g | 2x A100 80GB | 160GB | 24 |
| a2-ultragpu-4g | 4x A100 80GB | 320GB | 48 |
| a2-ultragpu-8g | 8x A100 80GB | 640GB | 96 |

SkyPilot picks the right machine type based on `accelerators: A100:4` (40GB) or `A100-80GB:4` (80GB).

## Tested

| Instance | GPUs | Model | Config | Result |
|---|---|---|---|---|
| a2-highgpu-4g (spot) | 4x A100 40GB | Qwen3-0.6B | TP=4 | Working (europe-west4) |

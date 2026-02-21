# Azure — NC96ads_A100_v4 (4x A100 PCIe 80GB)

No pre-built image. Run install script on a fresh VM (~15 min).

## Launch + Install

```bash
# 1. Create resource group
az group create --name vllm-rg --location westus3

# 2. Create VM
#    IMPORTANT: --security-type Standard disables Secure Boot (required for NVIDIA .run installer)
az vm create \
  --resource-group vllm-rg \
  --name a100-vllm \
  --size Standard_NC96ads_A100_v4 \
  --image Canonical:ubuntu-24_04-lts:server:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --os-disk-size-gb 200 \
  --security-type Standard

# 3. SSH in, copy and run install script
scp build-azure-a100.sh azureuser@<PUBLIC_IP>:/tmp/
ssh azureuser@<PUBLIC_IP>
sudo bash /tmp/build-azure-a100.sh

# 4. Activate and serve
source /opt/vllm-env/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model <model> --tensor-parallel-size 4 --port 8000

# 5. Query
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"<model>","prompt":"Hello","max_tokens":100}'
```

## Differences from AWS

| | AWS (p4d) | Azure (NC96) |
|---|---|---|
| GPUs | 8x A100 40GB SXM4 | 4x A100 80GB PCIe |
| Interconnect | NVSwitch (600 GB/s) | PCIe Gen4 (64 GB/s) |
| Fabric Manager | Required | Not needed |
| Max TP | 8 | 4 |
| Secure Boot | Not an issue | Must disable at creation |

## Verification

```bash
nvidia-smi                                    # 4x A100 PCIe 80GB, driver 580.105.08
systemctl status nvidia-persistenced           # active
nvcc --version                                # 12.8
source /opt/vllm-env/bin/activate
python -c "import torch; print(torch.cuda.device_count())"   # 4
python -c "import vllm; print(vllm.__version__)"             # 0.10.0
```

## Cleanup

Delete everything in one shot:
```bash
az group delete --name vllm-rg --yes
```

## Quota (approved)

| VM Series | Region | Cores |
|---|---|---|
| NC A100 v4 | westus3 | 100 |
| NCasT4v3 | westus | 100 |
| NCasT4v3 | eastus | 100 |
| NCH100v5 | centralus | 80 |

## Tested

| Model | Config | Result |
|---|---|---|
| Qwen2.5-72B-Instruct | TP=4 | 1000 tokens at 25 tok/s |

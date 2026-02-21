# AWS — p4d.24xlarge (8x A100 SXM4 40GB)

## Pre-built AMI (ready to use)

**AMI**: `ami-04f8546cd7cc1dcd9` | Region: `us-east-1` | Ubuntu 24.04

```bash
# 1. Launch
aws ec2 run-instances --image-id ami-04f8546cd7cc1dcd9 --instance-type p4d.24xlarge \
  --key-name <your-key> --security-group-ids <your-sg>

# 2. SSH in, activate, serve
source /opt/vllm-env/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model <model> --tensor-parallel-size 4 --port 8000

# 3. Query
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"<model>","prompt":"Hello","max_tokens":100}'
```

Everything is pre-installed. Just launch, activate, serve.

## Copy AMI to another region

```bash
aws ec2 copy-image --source-region us-east-1 --source-image-id ami-04f8546cd7cc1dcd9 \
  --region <target-region> --name "p4d-vllm010-cuda128"
```

## Rebuild AMI from scratch

1. Launch p4d.24xlarge with base Ubuntu 24.04 AMI `ami-0071174ad8cbb9e17` (us-east-1).
   **Set `DeleteOnTermination=false`** on root EBS volume.
2. `scp build-p4d-ami.sh` to the instance.
3. `sudo bash build-p4d-ami.sh` — takes ~15 min.
4. Verify: `nvidia-smi` shows 8x A100, both services active, vllm imports.
5. Stop instance, then create image:
   ```bash
   aws ec2 create-image --instance-id <id> --name "p4d-vllm010-cuda128-$(date +%Y%m%d)"
   ```

## Verification

```bash
nvidia-smi                                    # 8x A100, driver 580.105.08
systemctl status nvidia-fabricmanager          # active
systemctl status nvidia-persistenced           # active
nvcc --version                                # 12.8
source /opt/vllm-env/bin/activate
python -c "import torch; print(torch.cuda.device_count())"   # 8
python -c "import vllm; print(vllm.__version__)"             # 0.10.0
```

## Quota

- p4d.24xlarge = 96 vCPUs from "Running On-Demand P instances" quota (L-417A185B).
- Current limit: 128 (1 instance). Increase to 192+ pending for 2-node tests.
- Check: `aws service-quotas get-service-quota --service-code ec2 --quota-code L-417A185B`

## Tested

| Model | Config | Result |
|---|---|---|
| Qwen3-0.6B | TP=4 | Completions, chat, streaming all working |

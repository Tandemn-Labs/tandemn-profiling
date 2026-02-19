# Azure + SkyPilot Setup

## Prerequisites (one-time)

### 1. Install Azure CLI

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### 2. Install SkyPilot with Azure support

```bash
# If using uv (azure-cli needs pre-release deps):
uv pip install --prerelease=allow "skypilot-nightly[azure]"

# If using pip (no extra flags needed):
pip install "skypilot-nightly[azure]"
```

### 3. Login and set subscription

```bash
az login

# List subscriptions to find the one with GPU quota
az account list -o table

# Set it
az account set -s <YOUR_SUBSCRIPTION_ID>
```

### 4. Register resource providers

Azure doesn't auto-register all providers on new subscriptions. SkyPilot needs
these to create VMs, networking, and storage. Without them `sky launch` fails
with cryptic errors.

```bash
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.Storage
```

Verify all three say `Registered`:

```bash
az provider show -n Microsoft.Compute -o table
az provider show -n Microsoft.Network -o table
az provider show -n Microsoft.Storage -o table
```

### 5. Verify SkyPilot sees Azure

```bash
sky check azure
```

You should see a green checkmark for Azure. No service principal, resource
group, or storage account needed — SkyPilot creates those automatically on
first launch.

---

## Approved GPU Quota

| GPU  | Azure SKU      | Region      | vCPUs |
|------|----------------|-------------|-------|
| T4   | NCasT4v3       | eastus      | 100   |
| T4   | NCasT4v3       | westus      | 100   |
| H100 | NCH100v5       | centralus   | 80    |
| A100 | NCA100v4       | westus3     | 100   |

---

## Step 1: On-Demand Launch

From the repo root (`tandemn-profiling/`):

```bash
# Dry run first — see cost estimate without launching
sky launch roofline/azure/azure-demo-t4-ondemand.yaml --dryrun

# Launch the demo on Azure (on-demand T4 in eastus)
sky launch roofline/azure/azure-demo-t4-ondemand.yaml

# Check status
sky status

# Stream logs
sky logs azure-demo-t4

# SSH into the instance
sky ssh azure-demo-t4

# When done — tear down to stop billing
sky down azure-demo-t4
```

This will:
1. Provision a T4 VM in Azure eastus
2. Install vLLM + dependencies
3. Run 50 batched summarization requests with Qwen3-0.6B
4. Print throughput metrics (tokens/s, TTFT, requests/s)

### Download results

```bash
sky rsync-down azure-demo-t4 /tmp/azure_demo_results.json ./results/
```

---

## Step 2: Spot Instances (after on-demand works)

Change one line in the YAML:

```yaml
resources:
  use_spot: true
```

Or create a separate YAML (already provided):

```bash
sky launch roofline/azure/azure-demo-t4-spot.yaml
```

SkyPilot handles spot preemption + automatic recovery.

---

## Step 3: SkyServe (after spot works)

Wrap the model in an HTTP server and use `sky serve`:

```bash
sky serve up roofline/azure/azure-serve.yaml
```

This gives you:
- Auto-scaling replicas
- Load balancing
- Spot + on-demand fallback

---

## Useful Commands

```bash
# List all running clusters
sky status

# SSH into a running cluster
sky ssh azure-demo-t4

# Stream logs from a running/finished job
sky logs azure-demo-t4

# Tear down everything
sky down -a
```

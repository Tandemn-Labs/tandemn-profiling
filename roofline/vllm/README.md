# Batch inference benchmarking with vLLM


Benchmarking framework for profiling distributed LLM inference with [vLLM](https://github.com/vllm-project/vllm), systematically evaluating how different combinations of **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)** affect throughput, latency, GPU utilization, and cost-efficiency across GPU types.

**Models:** `Qwen/Qwen3-32B`, `Qwen/Qwen2.5-72B-Instruct`, `Qwen/Qwen3-235B-A22B`
**Dataset:** `emozilla/pg19-test` (Project Gutenberg books, used as summarization prompts)

---

## Architecture

```
┌──────────────────────┐
│  automatic_benchmark  │  Main entrypoint: define sweep (models × TP/PP × IO lengths × GPU)
│        _1.py          │  → generates experiment CSV, then calls automatic_launch_1.py
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  automatic_launch     │  For each row in the CSV:
│       _1.py           │    1. Select GPU instance type & node count
│                       │    2. Provision cluster via SkyPilot
│                       │    3. Launch benchmark script
│                       │    4. Collect results & tear down
└─────────┬────────────┘
          │  provisions
          ▼
┌─────────────────────────────────────────────────┐
│  SkyPilot Cluster (AWS)                         │
│  ┌────────────────────────────────────────────┐ │
│  │  roofline_benchmarks/                      │ │
│  │    roofline-tp{X}-pp{Y}-...yaml  (cluster) │ │
│  │    benchmark_roofline-tp{X}-pp{Y}-...py    │ │
│  │                                            │ │
│  │  1. Start Ray cluster across nodes         │ │
│  │  2. Launch vLLM with TP/PP config          │ │
│  │  3. GPU monitor threads (pynvml, 0.5s)     │ │
│  │  4. Scheduler monitor (0.25s)              │ │
│  │  5. Run 30 inference requests              │ │
│  │  6. Collect & save results                 │ │
│  └────────────────────────────────────────────┘ │
└────────┬────────────────────────────────────────┘
         │  outputs
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  results/            │────▶│  plot_benchmark      │
│   results.json       │     │    _results.py       │
│   timeseries_*.json  │     │  plot_timeseries.py  │
│   benchmark.log      │     │       → PDF charts   │
│   *_merged.csv       │     └──────────────────────┘
└─────────────────────┘
```

### Key components

| File | Purpose |
|---|---|
| `automatic_benchmark_1.py` | **Main entrypoint.** Defines the full sweep (models × TP/PP × IO lengths × GPU types), generates the experiment CSV, and kicks off `automatic_launch_1.py` |
| `automatic_launch_1.py` | Orchestrates the full pipeline: selects instance types, provisions SkyPilot clusters, runs benchmarks, collects results, tears down |
| `benchmark_client.py` | Async HTTP client that sends inference requests to a running vLLM server and measures throughput/latency |
| `plot_benchmark_results.py` | Bar charts comparing throughput, GPU utilization, and cost-efficiency across TP/PP configs |
| `plot_timeseries.py` | Time-series plots of per-GPU SM%, memory%, memory bandwidth% during a benchmark run |

### GPU monitoring

- **Single-node:** A background thread uses `pynvml` to sample GPU SM utilization, memory usage, and memory bandwidth every 0.5s.
- **Multi-node:** Ray actors are deployed one-per-node via `STRICT_SPREAD` placement groups, each running the same pynvml sampling loop. Results are merged by timestamp after the run.
- **Scheduler monitoring:** A separate thread polls the vLLM scheduler every 0.25s to record queue depths (running, waiting, swapped) and KV cache utilization.

### VM selection strategy

`automatic_launch_1.py` selects instance types using two strategies:
- **`prefer_single_node`** — fits all GPUs (TP × PP) into a single instance when possible, minimizing inter-node latency.
- **`fit_tp_then_scale`** — ensures the TP degree fits within a single node's GPUs, then adds nodes for PP.

### Tested hardware

| GPU | AWS Instance | GPUs/node | VRAM | vLLM version |
|---|---|---|---|---|
| L4 | g6.12xlarge / g6.48xlarge | 4 / 8 | 24GB | 0.10.0 |
| L40S | g6e.12xlarge / g6e.48xlarge | 4 / 8 | 48GB | 0.10.0 |
| A10G | g5.12xlarge / g5.48xlarge | 4 / 8 | 24GB | 0.10.0 |
| A100-40GB | p4d.24xlarge | 8 | 40GB | 0.10.0 (AMI: driver 580, CUDA 12.8) |
| H100 | p5.48xlarge | 8 | 80GB | 0.10.0 |

---

## How to run

### Prerequisites

- Python 3.12+
- [SkyPilot](https://skypilot.readthedocs.io/) configured with AWS credentials (`sky check`)
- Install dependencies:

```bash
pip install skypilot-nightly aiohttp requests numpy pandas boto3
```

### 1. Dry-run (preview the plan)

Always do this first to verify what will be launched and the estimated cost:

```bash
cd roofline/vllm

python3 automatic_benchmark_1.py                        # all GPU types
python3 automatic_benchmark_1.py --gpu L40S             # L40S only
python3 automatic_benchmark_1.py --gpu L4 --gpu L40S    # multiple GPU types
```

### 2. Launch the benchmarks

```bash
# Full sweep on one GPU type
python3 automatic_benchmark_1.py --gpu L40S --run

# Specific model only
python3 automatic_benchmark_1.py --gpu L4 --run --models Qwen/Qwen3-32B

# Specific TP/PP + IO shapes
python3 automatic_benchmark_1.py --gpu L4 --run --tp-pp 8,3 --io 512,1024 1024,512

# Use S3 model weights instead of HuggingFace
python3 automatic_benchmark_1.py --gpu L40S --run --s3-models

# GCP instead of AWS
python3 automatic_benchmark_1.py --gpu L4 --run --cloud gcp
```

`automatic_benchmark_1.py` generates the experiment CSV and calls `automatic_launch_1.py`, which for each experiment:
1. Selects the cheapest matching instance configuration for the required GPU count (`TP × PP`).
2. Provisions the cluster with `sky launch`, starts Ray, and runs the benchmark.
3. Downloads `results.json`, `timeseries_*.json`, `benchmark_results-*.csv`, and `benchmark.log` into `results/`.
4. Tears down the cluster.
5. Sends progress notifications to Discord (webhook configured in `automatic_launch_1.py`).

### 3. Plot the results

**Bar-chart comparison across TP/PP configs:**

```bash
python plot_benchmark_results.py
```

Reads `benchmark_results_merged.csv` and outputs a PDF with three subplots:
1. **Throughput** — requests/sec, input/output/total tokens/sec (with min–max error bars)
2. **GPU utilization** — SM% and memory bandwidth%
3. **Cost efficiency** — tokens per dollar

**Time-series of a single run:**

```bash
python plot_timeseries.py
```

Reads a `timeseries_*.json` file and outputs a PDF with:
- Per-GPU SM utilization, memory utilization, and memory bandwidth over time
- Scheduler queue depths (running / waiting / swapped) and KV cache utilization

---

## Result format

Each benchmark run produces three files saved under `results/result-{input}in_{output}out/{cloud}-{gpu}/`:

### `results.json`

A JSON array with one object per run. Key fields:

```jsonc
[{
  // Experiment config
  "tp": 4,
  "pp": 2,
  "max_input_length": 2048,
  "max_output_length": 512,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
  "exp_id": "tp4_pp2_in2048_out512",

  // Performance
  "elapsed_time": 79.39,             // seconds
  "requests_per_sec": 0.378,
  "input_tokens_per_sec": 774.28,
  "output_tokens_per_sec": 192.72,
  "total_tokens_per_sec": 967.01,

  // Cost
  "instance_type": "g6e.48xlarge",
  "price_per_hour": 13.35,
  "cost_for_run_usd": 0.2944,
  "tokens_per_dollar": 260765.62,

  // GPU metrics (per-GPU)
  "gpu0_sm_pct_avg": 91.04,
  "gpu0_membw_pct_avg": 44.07,
  "gpu0_mem_gb_avg": 39.92,
  // ... gpu1 through gpu7 ...

  // Aggregates
  "avg_sm_util_pct": 92.95,
  "avg_mem_bw_util_pct": 44.68,

  // vLLM engine config (extracted at runtime)
  "llm_tensor_parallel_size": 4,
  "llm_pipeline_parallel_size": 2,
  "llm_max_model_len": 2559,
  "config_num_gpu_blocks": 32263,
  "config_dtype": "torch.bfloat16",

  "status": "success"
}]
```

### `timeseries_*.json`

Detailed time-series sampled during the run:

```jsonc
{
  "exp_id": "tp4_pp2_in2048_out512",
  "elapsed_time": 79.39,
  "gpu_timeseries": [
    // Sampled every ~0.5s
    { "t": 0.0,   "gpu0_sm_pct": 18, "gpu0_mem_gb": 39.5, "gpu0_membw_pct": 14, ... },
    { "t": 0.509, "gpu0_sm_pct": 100, "gpu0_mem_gb": 39.5, "gpu0_membw_pct": 3, ... },
    ...
  ],
  "scheduler_timeseries": [
    // Sampled every ~0.25s
    { "t": 0.0,  "running": 0, "waiting": 30, "swapped": 0, "kv_cache_util_pct": 5.2 },
    { "t": 0.25, "running": 8, "waiting": 22, "swapped": 0, "kv_cache_util_pct": 45.8 },
    ...
  ]
}
```

### `benchmark.log`

Full stdout/stderr from the benchmark process.

### `benchmark_results_merged.csv`

Aggregated CSV combining all successful runs in a result directory (150+ columns). Used as input for `plot_benchmark_results.py`.

---

## Directory structure

```
roofline/vllm/
├── automatic_benchmark_1.py        # Main entrypoint (sweep definition + launcher)
├── automatic_launch_1.py           # Cluster orchestrator (called by automatic_benchmark_1.py)
├── benchmark_client.py             # Async HTTP benchmark client
├── plot_benchmark_results.py       # Bar-chart visualizations
├── plot_timeseries.py              # Time-series visualizations
├── results/                        # Benchmark outputs per run
│   ├── aws_g6_L4/                  # L4 results
│   │   └── tp{X}-pp{Y}-...-success/
│   │       ├── benchmark_results-TIMESTAMP.csv
│   │       ├── results.json
│   │       ├── timeseries_*.json
│   │       └── benchmark.log
│   └── aws_g6e_L40S/               # L40S results (same structure)
├── perfdb_all.csv                  # Consolidated perf database (all runs)
└── failed/                         # Failed runs (logs + partial results)
```

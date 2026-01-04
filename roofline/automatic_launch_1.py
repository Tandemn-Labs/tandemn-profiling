import csv
import json
from dataclasses import dataclass
import subprocess
from collections import defaultdict
import sys
import argparse
from pathlib import Path
import requests
import signal
import atexit
import logging
from datetime import datetime

# GPU type configurations
# Each GPU type maps to: available GPU counts per instance, instance family, and pricing
GPU_CONFIGS = {
    "L40S": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g6e",
        "pricing": {
            1: {"instance_type": "g6e.2xlarge", "price_per_hour": 0.99},
            4: {"instance_type": "g6e.12xlarge", "price_per_hour": 4.68},
            8: {"instance_type": "g6e.48xlarge", "price_per_hour": 13.35},
        }
    },
    "L4": {
        "available_gpus": [1, 2, 4],
        "instance_family": "g6",
        "pricing": {
            1: {"instance_type": "g6.2xlarge", "price_per_hour": 0.526},
            2: {"instance_type": "g6.12xlarge", "price_per_hour": 0.752},
            4: {"instance_type": "g6.48xlarge", "price_per_hour": 1.204},
        }
    },
    "A10G": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g5",
        "pricing": {
            1: {"instance_type": "g5.2xlarge", "price_per_hour": 1.006},
            4: {"instance_type": "g5.12xlarge", "price_per_hour": 4.096},
            8: {"instance_type": "g5.48xlarge", "price_per_hour": 16.384},
        }
    },
    "A100-40gb": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p4d",
        "pricing": {
            1: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs, but can use 1
            4: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs, but can use 4
            8: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs (40GB per GPU)
        }
    },
    "A100-80gb": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p4de",
        "pricing": {
            1: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs, but can use 1
            4: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs, but can use 4
            8: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs (80GB per GPU)
        }
    },
    "H100": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p5",
        "pricing": {
            1: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs, but can use 1
            4: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs, but can use 4
            8: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs
        }
    },
}

# Default GPU type (for backward compatibility)
DEFAULT_GPU_TYPE = "L40S"

# Global logger instance
logger = logging.getLogger("benchmark")

def setup_logger(log_file_path: Path):
    """Set up logger to write to both console and file."""
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler - prints to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler - writes to log file
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Track active cluster for cleanup on unexpected exit
_active_cluster = None
_active_cluster_file = Path("/tmp/.active_benchmark_cluster")

def set_active_cluster(cluster_name):
    """Set active cluster and persist to file for crash recovery."""
    global _active_cluster
    _active_cluster = cluster_name
    if cluster_name:
        _active_cluster_file.write_text(cluster_name)
    elif _active_cluster_file.exists():
        _active_cluster_file.unlink()

def get_active_cluster():
    """Get active cluster from memory or file."""
    global _active_cluster
    if _active_cluster:
        return _active_cluster
    if _active_cluster_file.exists():
        return _active_cluster_file.read_text().strip()
    return None

def cleanup_on_exit():
    """Cleanup handler for unexpected exits."""
    cluster = get_active_cluster()
    if cluster:
        print(f"\n⚠️  Cleanup triggered. Tearing down cluster: {cluster}")
        try:
            subprocess.run(["sky", "down", "-y", cluster], timeout=300)
            print(f"✅ Cluster {cluster} terminated.")
        except Exception as e:
            print(f"❌ Failed to terminate cluster {cluster}: {e}")
            print(f"   Run manually: sky down -y {cluster}")
        finally:
            # Clear the file
            if _active_cluster_file.exists():
                _active_cluster_file.unlink()

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    print(f"\n🛑 Received signal {signum}. Cleaning up...")
    cleanup_on_exit()
    sys.exit(1)

def check_orphaned_cluster():
    """Check if there's an orphaned cluster from a previous crashed run."""
    if _active_cluster_file.exists():
        cluster = _active_cluster_file.read_text().strip()
        print(f"\n⚠️  Found orphaned cluster from previous run: {cluster}")
        print(f"   This cluster may still be running and costing money!")
        response = input(f"   Terminate it now? [Y/n]: ").strip().lower()
        if response != 'n':
            print(f"   Terminating {cluster}...")
            subprocess.run(["sky", "down", "-y", cluster])
            _active_cluster_file.unlink()
            print(f"   ✅ Done.")
        else:
            print(f"   Skipped. Run 'sky down -y {cluster}' manually when ready.")
            _active_cluster_file.unlink()

# Register cleanup handlers
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_cluster_config(tp, pp, gpu_type=DEFAULT_GPU_TYPE):
    """
    Calculate the best instance configuration for given TP/PP and GPU type.

    Returns (gpus_per_node, num_nodes) that can accommodate TP×PP GPUs.

    Examples:
        TP=4, PP=2 → 4 GPUs/node × 2 nodes = 8 GPUs (TP fits in one node)
        TP=2, PP=4 → 4 GPUs/node × 2 nodes = 8 GPUs (2 TP groups per node, 2 PP stages per node)
        TP=8, PP=1 → 8 GPUs/node × 1 node = 8 GPUs
    """
    if pp < 1:
        raise ValueError(f"Invalid PP={pp}. Must be >= 1")
    if tp < 1:
        raise ValueError(f"Invalid TP={tp}. Must be >= 1")
    if gpu_type not in GPU_CONFIGS:
        raise ValueError(f"Invalid GPU type: {gpu_type}. Must be one of {list(GPU_CONFIGS.keys())}")

    total_gpus = tp * pp
    available_gpus = GPU_CONFIGS[gpu_type]["available_gpus"]

    # Find the smallest instance type that can fit TP GPUs (for fast TP communication)
    gpus_per_node = None
    for gpu_count in available_gpus:
        if gpu_count >= tp:
            gpus_per_node = gpu_count
            break

    if gpus_per_node is None:
        raise ValueError(f"TP={tp} exceeds max GPUs per instance ({max(available_gpus)}) for {gpu_type}")

    # Calculate how many nodes we need
    num_nodes = (total_gpus + gpus_per_node - 1) // gpus_per_node  # Ceiling division

    # Make sure we have enough total GPUs
    actual_gpus = gpus_per_node * num_nodes
    if actual_gpus < total_gpus:
        raise ValueError(f"Cannot fit TP={tp}, PP={pp} ({total_gpus} GPUs) into available {gpu_type} instances")

    return gpus_per_node, num_nodes


def load_experiments(csv_path):
    experiments = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append({
                'tp': int(row['tensor_degree']),
                'pp': int(row['pipeline_degree']),
                'max_input_length': int(row['max_input_length']),
                'max_output_length': int(row['max_output_length']),
                'model': row['model']
            })
    return experiments

def group_by_cluster(experiments, gpu_type=DEFAULT_GPU_TYPE):
    groups = defaultdict(list)
    for exp in experiments:
        gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'], gpu_type)
        key = (gpus_per_node, num_nodes)
        groups[key].append(exp)
    return dict(groups)

def group_by_input_output_then_cluster(experiments, gpu_type=DEFAULT_GPU_TYPE):
    """Group experiments first by input/output length, then by cluster config."""
    # First level: group by input/output length
    io_groups = defaultdict(list)
    for exp in experiments:
        io_key = (exp['max_input_length'], exp['max_output_length'])
        io_groups[io_key].append(exp)
    
    # Second level: within each IO group, group by cluster config
    result = {}
    for io_key, io_exps in io_groups.items():
        cluster_groups = defaultdict(list)
        for exp in io_exps:
            gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'], gpu_type)
            cluster_key = (gpus_per_node, num_nodes)
            cluster_groups[cluster_key].append(exp)
        result[io_key] = dict(cluster_groups)
    
    return result

def generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type=DEFAULT_GPU_TYPE):
    lmcache_exports = """
  # LMCache environment variables (must be set before Ray workers start)
  export LMCACHE_USE_EXPERIMENTAL="True"
  export LMCACHE_CHUNK_SIZE="256"
  export LMCACHE_LOCAL_CPU="True"
  export LMCACHE_MAX_LOCAL_CPU_SIZE="40.0"
  export LMCACHE_SAVE_UNFULL_CHUNK="True"
  # export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE="shm"
  # export VLLM_USE_RAY_WRAPPED_PP_COMM=1
  export LMCACHE_ENABLE_ASYNC_LOADING="False"
  export LMCACHE_REMOTE_SERDE="cachegen"
  export LMCACHE_USE_LAYERWISE="True"
  export LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR="True"
  export NCCL_P2P_DISABLE=1
  export NCCL_TIMEOUT=3600
  export TORCH_NCCL_BLOCKING_WAIT=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_DEBUG=INFO
  # export LMCACHE_LOOKUP_TIMEOUT_MS="12000"
  # export LMCACHE_LOCAL_DISK="/tmp/lmcache_disk"
  # export LMCACHE_MAX_LOCAL_DISK_SIZE="100"
  # export LMCACHE_DISK_PERSISTENCE="True"
  # export LMCACHE_LOG_LEVEL="INFO"
"""

    # Check if any experiment needs Ray (PP > 1)
    needs_ray = any(exp['pp'] > 1 for exp in experiments)

    ray_run = f"""{lmcache_exports}
  python roofline_benchmarks/benchmark_{cluster_name}.py
    """
    if num_nodes > 1:
        # Multi-node: use Sky's Ray cluster startup script
        ray_run = f"""{lmcache_exports}
  # Start Ray cluster across all nodes
  export RAY_CMD="uv run ray"
  export RAY_DASHBOARD_HOST=0.0.0.0
  export RAY_HEAD_PORT=6379
  export RAY_DASHBOARD_PORT=8265
  export RAY_CGRAPH_get_timeout=1800
  export RAY_CGRAPH_submit_timeout=180
  ~/sky_templates/ray/start_cluster

  if [ "${{SKYPILOT_NODE_RANK}}" = "0" ]; then
      echo "Ray cluster started on head node"
      ray status --address="127.0.0.1:${{RAY_HEAD_PORT}}" || true

      # Run benchmark here while Ray is still up
      PYTHONHASHSEED=0 python roofline_benchmarks/benchmark_{cluster_name}.py
  fi
        """
    elif needs_ray:
        # Single-node but needs Ray for pipeline parallelism
        ray_run = f"""{lmcache_exports}
  # Start Ray on single node for pipeline parallelism
  export RAY_DASHBOARD_HOST=0.0.0.0
  export RAY_HEAD_PORT=6379
  export RAY_DASHBOARD_PORT=8265

  echo "Starting Ray for single-node pipeline parallelism..."
  uv run ray start --head --port=${{RAY_HEAD_PORT}} --dashboard-host=${{RAY_DASHBOARD_HOST}} --dashboard-port=${{RAY_DASHBOARD_PORT}} --num-cpus=0

  sleep 5
  uv run ray status --address="127.0.0.1:${{RAY_HEAD_PORT}}"

  # Run benchmark
  PYTHONHASHSEED=0 python roofline_benchmarks/benchmark_{cluster_name}.py

  # Cleanup Ray
  uv run ray stop
        """
    # Handle A100 variants: use A100 as accelerator name but specify instance type
    accelerator_name = gpu_type
    instance_type_constraint = ""
    if gpu_type in ["A100-40gb", "A100-80gb"]:
        accelerator_name = "A100"
        # Force specific instance type (p4d for 40gb, p4de for 80gb)
        gpu_config = GPU_CONFIGS[gpu_type]
        instance_type = gpu_config["pricing"][gpus_per_node]["instance_type"]
        instance_type_constraint = f"  instance_type: {instance_type}\n"
    
    return f"""
name: {cluster_name}
resources:
  cloud: aws
  accelerators: {accelerator_name}:{gpus_per_node}
{instance_type_constraint}  use_spot: false
  disk_size: 500GB
  memory: "64GB+"
  # No region constraint - SkyPilot will try all available AWS regions
  # This helps find capacity when us-east-1 is full (especially during US business hours)
num_nodes: {num_nodes}
workdir: .
setup: | 
  export PYTHONHASHSEED=0
  set -euxo pipefail
  python3 -m pip install -U pip
  python3 -m pip install -U uv

  uv venv --python 3.12 --seed
  source .venv/bin/activate

  # Install vLLM and LMCache

  uv pip install "datasets" "requests" "pynvml"
  uv pip install "vllm==0.10.0" # old vllm
  # uv pip install "lmcache==0.3.11"
  git clone https://github.com/lmcache/lmcache.git
  cd lmcache
  git checkout v0.3.6
  uv pip install . --no-build-isolation
  cd ..

  mkdir -p /tmp/lmcache_disk

run: |
  set -euxo pipefail
  source .venv/bin/activate {ray_run}

  echo "Cluster ready for benchmarking"
"""

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453154642706960485/iFXIAaDTLxNO7_GHKHhXnXwFFnXziniP4TUwLUDUnXHtT9kNo08eQBjGQ4CiBr6AazY6"

def send_discord_message(message):
    payload = {"content": message}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)

def generate_benchmark_script(experiments, gpus_per_node, num_nodes, gpu_type=DEFAULT_GPU_TYPE):
    """Generate Python benchmark script for a set of experiments."""
    exp_list = json.dumps(experiments, indent=2)

    # Calculate cluster pricing info
    gpu_config = GPU_CONFIGS[gpu_type]
    price_per_node = gpu_config["pricing"][gpus_per_node]["price_per_hour"]
    total_price_per_hour = price_per_node * num_nodes
    instance_type = gpu_config["pricing"][gpus_per_node]["instance_type"]
    if num_nodes > 1:
        cluster_instance_type = f"{num_nodes}x {instance_type}"
    else:
        cluster_instance_type = instance_type
    
    return f'''#!/usr/bin/env python3
import os
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
os.environ["LMCACHE_LOG_LEVEL"] = "INFO"

import requests
import json
import time
import torch
import gc
import ray
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed import cleanup_dist_env_and_memory
import shutil
import subprocess
import time
import threading
from collections import defaultdict

# ============================================================================
# Distributed GPU Monitoring with Ray Actors
# ============================================================================

@ray.remote
class GPUMonitorActor:
    """Ray actor for GPU monitoring on a single node."""

    def __init__(self, node_id: str, sample_interval: float = 0.5):
        self.node_id = node_id
        self.sample_interval = sample_interval
        self.timeseries = []
        self._start_time = None
        self._stop_event = None
        self._pynvml_available = False
        self._pynvml = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
            self._device_count = pynvml.nvmlDeviceGetCount()
            print(f"[{{self.node_id}}] GPU monitor initialized with {{self._device_count}} GPUs")
        except Exception as e:
            print(f"[{{self.node_id}}] pynvml not available: {{e}}")
            self._device_count = 0

    def start(self):
        """Start collecting samples."""
        if not self._pynvml_available:
            return
        self.timeseries = []
        self._start_time = time.time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop collecting samples."""
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        if hasattr(self, '_thread') and self._thread is not None:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        pynvml = self._pynvml
        while not self._stop_event.is_set():
            timestamp = time.time()
            relative_time = timestamp - self._start_time
            sample = {{'t': round(relative_time, 3)}}

            for i in range(self._device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_util_pct = (mem_info.used / mem_info.total) * 100

                    # GPU utilization (SM utilization)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                    mem_bw_util_pct = util.memory

                    # Use node-prefixed keys
                    sample[f'{{self.node_id}}_gpu{{i}}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'{{self.node_id}}_gpu{{i}}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'{{self.node_id}}_gpu{{i}}_sm_pct'] = gpu_util_pct
                    sample[f'{{self.node_id}}_gpu{{i}}_membw_pct'] = mem_bw_util_pct
                except Exception:
                    pass

            if len(sample) > 1:
                self.timeseries.append(sample)

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return collected time-series data."""
        return self.timeseries

    def get_node_id(self):
        """Return this actor's node ID."""
        return self.node_id

    def get_device_count(self):
        """Return number of GPUs on this node."""
        return self._device_count


class DistributedGPUMonitor:
    """Manage GPU monitor actors across all Ray nodes."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.actors = []
        self.node_map = {{}}  # node_id -> actor

    def start(self):
        """Deploy actors on all Ray nodes and start monitoring."""
        # Get all Ray nodes
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n.get('Alive', False)]

        print(f"📡 Found {{len(alive_nodes)}} Ray nodes for GPU monitoring")

        for idx, node in enumerate(alive_nodes):
            node_id = f"node{{idx}}"
            node_ip = node.get('NodeManagerAddress', 'unknown')

            # Create actor pinned to this specific node
            resources = {{f"node:{{node.get('NodeID')}}": 0.001}}
            try:
                actor = GPUMonitorActor.options(
                    resources=resources,
                    num_cpus=0.1,  # Minimal CPU
                ).remote(node_id, self.sample_interval)

                self.actors.append(actor)
                self.node_map[node_id] = actor
                print(f"  ✓ Deployed monitor on {{node_id}} ({{node_ip}})")
            except Exception as e:
                print(f"  ✗ Failed to deploy on {{node_id}}: {{e}}")

        # Start all actors
        ray.get([actor.start.remote() for actor in self.actors])
        print(f"📊 GPU monitoring started on {{len(self.actors)}} nodes")

    def stop(self):
        """Stop all monitor actors."""
        if self.actors:
            ray.get([actor.stop.remote() for actor in self.actors])

    def get_timeseries(self):
        """Collect and merge time-series from all nodes."""
        if not self.actors:
            return []

        # Collect from all actors
        all_series = ray.get([actor.get_timeseries.remote() for actor in self.actors])

        # Merge time-series by timestamp
        merged = {{}}
        for series in all_series:
            for sample in series:
                t = sample['t']
                if t not in merged:
                    merged[t] = {{'t': t}}
                for key, value in sample.items():
                    if key != 't':
                        merged[t][key] = value

        # Sort by timestamp and return as list
        return [merged[t] for t in sorted(merged.keys())]

    def get_summary(self):
        """Return summary statistics across all nodes."""
        timeseries = self.get_timeseries()
        if not timeseries:
            return {{}}

        summary = {{}}
        metrics = defaultdict(list)

        for sample in timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        # Per-GPU summaries
        for key, values in metrics.items():
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = round(max(values), 2)
                summary[f'{{key}}_min'] = round(min(values), 2)

        # Aggregate across all GPUs
        all_sm_util = []
        all_mem_bw_util = []
        all_mem_util = []
        for key, values in metrics.items():
            if '_sm_pct' in key:
                all_sm_util.extend(values)
            elif '_membw_pct' in key:
                all_mem_bw_util.extend(values)
            elif '_mem_pct' in key:
                all_mem_util.extend(values)

        if all_sm_util:
            summary['avg_sm_util_pct'] = round(sum(all_sm_util) / len(all_sm_util), 2)
            summary['max_sm_util_pct'] = round(max(all_sm_util), 2)
        if all_mem_bw_util:
            summary['avg_mem_bw_util_pct'] = round(sum(all_mem_bw_util) / len(all_mem_bw_util), 2)
            summary['max_mem_bw_util_pct'] = round(max(all_mem_bw_util), 2)
        if all_mem_util:
            summary['avg_mem_util_pct'] = round(sum(all_mem_util) / len(all_mem_util), 2)
            summary['max_mem_util_pct'] = round(max(all_mem_util), 2)

        summary['gpu_samples'] = len(timeseries)
        summary['num_nodes_monitored'] = len(self.actors)
        return summary


# ============================================================================
# Local GPU Monitoring (fallback for single-node)
# ============================================================================

# GPU Monitoring class using pynvml
class GPUMonitor:
    """Background GPU metrics collector using pynvml."""

    def __init__(self, sample_interval=0.5):
        self.sample_interval = sample_interval
        self.timeseries = []  # List of {{timestamp, gpu0_*, gpu1_*, ...}}
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._pynvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
        except Exception as e:
            print(f"⚠️  pynvml not available, GPU monitoring disabled: {{e}}")

    def start(self):
        if not self._pynvml_available:
            return
        self._stop_event.clear()
        self.timeseries = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _monitor_loop(self):
        pynvml = self._pynvml
        device_count = pynvml.nvmlDeviceGetCount()

        while not self._stop_event.is_set():
            timestamp = time.time()
            relative_time = timestamp - self._start_time
            sample = {{'t': round(relative_time, 3)}}

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_util_pct = (mem_info.used / mem_info.total) * 100

                    # GPU utilization (SM utilization)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu  # SM utilization
                    mem_bw_util_pct = util.memory  # Memory bandwidth utilization

                    sample[f'gpu{{i}}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'gpu{{i}}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'gpu{{i}}_sm_pct'] = gpu_util_pct
                    sample[f'gpu{{i}}_membw_pct'] = mem_bw_util_pct

                except Exception:
                    pass

            if len(sample) > 1:  # More than just timestamp
                self.timeseries.append(sample)

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return raw time-series data for plotting."""
        return self.timeseries

    def get_summary(self):
        """Return summary statistics of GPU metrics."""
        if not self.timeseries:
            return {{}}

        summary = {{}}

        # Collect values per metric
        metrics = defaultdict(list)
        for sample in self.timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        # Per-GPU summaries
        for key, values in metrics.items():
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = round(max(values), 2)
                summary[f'{{key}}_min'] = round(min(values), 2)

        # Aggregate across all GPUs
        all_sm_util = []
        all_mem_bw_util = []
        all_mem_util = []
        for key, values in metrics.items():
            if '_sm_pct' in key:
                all_sm_util.extend(values)
            elif '_membw_pct' in key:
                all_mem_bw_util.extend(values)
            elif '_mem_pct' in key:
                all_mem_util.extend(values)

        if all_sm_util:
            summary['avg_sm_util_pct'] = round(sum(all_sm_util) / len(all_sm_util), 2)
            summary['max_sm_util_pct'] = round(max(all_sm_util), 2)
        if all_mem_bw_util:
            summary['avg_mem_bw_util_pct'] = round(sum(all_mem_bw_util) / len(all_mem_bw_util), 2)
            summary['max_mem_bw_util_pct'] = round(max(all_mem_bw_util), 2)
        if all_mem_util:
            summary['avg_mem_util_pct'] = round(sum(all_mem_util) / len(all_mem_util), 2)
            summary['max_mem_util_pct'] = round(max(all_mem_util), 2)

        summary['gpu_samples'] = len(self.timeseries)
        return summary


class SchedulerMonitor:
    """Monitor vLLM scheduler queue depths and KV cache during generation."""

    def __init__(self, llm, sample_interval=0.25):
        self.llm = llm
        self.sample_interval = sample_interval
        self.timeseries = []
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self.timeseries = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _get_scheduler_metrics(self, engine, sample):
        """Extract scheduler queue depths from various vLLM versions."""
        scheduler = getattr(engine, 'scheduler', None)
        if scheduler is None:
            return

        # vLLM v0.10+ uses a list of schedulers (one per virtual engine)
        if isinstance(scheduler, list):
            total_running = 0
            total_waiting = 0
            total_swapped = 0
            for sched in scheduler:
                if hasattr(sched, 'running'):
                    total_running += len(sched.running)
                if hasattr(sched, 'waiting'):
                    total_waiting += len(sched.waiting)
                if hasattr(sched, 'swapped'):
                    total_swapped += len(sched.swapped)
            sample['running'] = total_running
            sample['waiting'] = total_waiting
            sample['swapped'] = total_swapped
        # Single scheduler (older vLLM versions)
        elif hasattr(scheduler, 'running'):
            sample['running'] = len(scheduler.running)
            if hasattr(scheduler, 'waiting'):
                sample['waiting'] = len(scheduler.waiting)
            if hasattr(scheduler, 'swapped'):
                sample['swapped'] = len(scheduler.swapped)

    def _get_kv_cache_metrics(self, engine, sample):
        """Extract KV cache utilization from vLLM."""
        # Try to access cache engine / block manager
        # vLLM stores KV cache info in different places depending on version

        # Method 1: scheduler[0].block_manager (vLLM v0.10+)
        scheduler = getattr(engine, 'scheduler', None)
        if scheduler is not None:
            scheds = scheduler if isinstance(scheduler, list) else [scheduler]
            for sched in scheds:
                block_mgr = getattr(sched, 'block_manager', None)
                if block_mgr is not None:
                    # Try to get GPU block usage
                    if hasattr(block_mgr, 'get_num_free_gpu_blocks'):
                        free_gpu = block_mgr.get_num_free_gpu_blocks()
                        total_gpu = getattr(block_mgr, 'num_total_gpu_blocks',
                                          getattr(block_mgr, 'num_gpu_blocks', None))
                        if total_gpu is not None:
                            used_gpu = total_gpu - free_gpu
                            sample['kv_cache_used_blocks'] = used_gpu
                            sample['kv_cache_total_blocks'] = total_gpu
                            sample['kv_cache_util_pct'] = round(100 * used_gpu / total_gpu, 1) if total_gpu > 0 else 0
                        break
                    # Alternative: gpu_allocator
                    gpu_alloc = getattr(block_mgr, 'gpu_allocator', None)
                    if gpu_alloc is not None:
                        if hasattr(gpu_alloc, 'get_num_free_blocks'):
                            free_gpu = gpu_alloc.get_num_free_blocks()
                            total_gpu = getattr(gpu_alloc, 'num_blocks', None)
                            if total_gpu is not None:
                                used_gpu = total_gpu - free_gpu
                                sample['kv_cache_used_blocks'] = used_gpu
                                sample['kv_cache_total_blocks'] = total_gpu
                                sample['kv_cache_util_pct'] = round(100 * used_gpu / total_gpu, 1) if total_gpu > 0 else 0
                            break

        # Method 2: cache_engine (some versions)
        cache_engine = getattr(engine, 'cache_engine', None)
        if cache_engine is not None and 'kv_cache_used_blocks' not in sample:
            # Try various attributes
            for attr in ['gpu_cache', 'cache']:
                cache = getattr(cache_engine, attr, None)
                if cache is not None and hasattr(cache, '__len__'):
                    sample['kv_cache_layers'] = len(cache)
                    break

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                timestamp = time.time()
                relative_time = timestamp - self._start_time
                sample = {{'t': round(relative_time, 3)}}

                engine = self.llm.llm_engine

                # Get scheduler queue depths
                self._get_scheduler_metrics(engine, sample)

                # Get KV cache utilization
                self._get_kv_cache_metrics(engine, sample)

                # Try to get number of unfinished requests
                if hasattr(engine, '_request_tracker'):
                    tracker = engine._request_tracker
                    if hasattr(tracker, 'get_num_unfinished_requests'):
                        sample['unfinished'] = tracker.get_num_unfinished_requests()

                if len(sample) > 1:  # More than just timestamp
                    self.timeseries.append(sample)

            except Exception as e:
                # Log first error for debugging
                if not self.timeseries:
                    print(f"⚠️  SchedulerMonitor error: {{e}}")

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return raw time-series data for plotting."""
        return self.timeseries

    def get_summary(self):
        """Return summary statistics of scheduler queue depths."""
        if not self.timeseries:
            return {{}}

        summary = {{}}

        # Collect all unique keys
        all_keys = set()
        for sample in self.timeseries:
            all_keys.update(sample.keys())
        all_keys.discard('t')

        for key in all_keys:
            values = [s.get(key, 0) for s in self.timeseries if key in s]
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = max(values)

        summary['scheduler_samples'] = len(self.timeseries)
        return summary

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453154642706960485/iFXIAaDTLxNO7_GHKHhXnXwFFnXziniP4TUwLUDUnXHtT9kNo08eQBjGQ4CiBr6AazY6"

def send_discord_message(message):
    payload = {{"content": message}}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)

# Clean up existing cache directory
cache_dir = "/tmp/lmcache_disk"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

os.makedirs(cache_dir, exist_ok=True)

EXPERIMENTS = {exp_list}
RESULTS_FILE = "/tmp/benchmark_results.json"
NUM_SAMPLES = 30

# Cluster pricing information
INSTANCE_TYPE = "{cluster_instance_type}"
PRICE_PER_HOUR = {total_price_per_hour}
NUM_NODES = {num_nodes}
GPUS_PER_NODE = {gpus_per_node}


def get_vllm_config_info(llm):
    """Extract configuration info from vLLM engine."""
    config_info = {{}}
    try:
        # Access the LLM engine's configuration
        engine = llm.llm_engine

        # Cache config (KV cache settings)
        if hasattr(engine, 'cache_config'):
            cache_cfg = engine.cache_config
            config_info['block_size'] = getattr(cache_cfg, 'block_size', None)
            config_info['gpu_memory_utilization'] = getattr(cache_cfg, 'gpu_memory_utilization', None)
            config_info['num_gpu_blocks'] = getattr(cache_cfg, 'num_gpu_blocks', None)
            config_info['num_cpu_blocks'] = getattr(cache_cfg, 'num_cpu_blocks', None)

        # Scheduler config (concurrency settings)
        if hasattr(engine, 'scheduler_config'):
            sched_cfg = engine.scheduler_config
            config_info['max_num_seqs'] = getattr(sched_cfg, 'max_num_seqs', None)
            config_info['max_num_batched_tokens'] = getattr(sched_cfg, 'max_num_batched_tokens', None)
            config_info['max_model_len'] = getattr(sched_cfg, 'max_model_len', None)
            config_info['chunked_prefill_enabled'] = getattr(sched_cfg, 'chunked_prefill_enabled', None)

        # Model config
        if hasattr(engine, 'model_config'):
            model_cfg = engine.model_config
            config_info['dtype'] = str(getattr(model_cfg, 'dtype', None))

    except Exception as e:
        print(f"⚠️  Could not extract vLLM config: {{e}}")

    return config_info

def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\\n{{'='*70}}")
    print(f"Running: TP={{exp['tp']}}, PP={{exp['pp']}}, "
          f"input={{exp['max_input_length']}}, output={{exp['max_output_length']}}")
    print("="*70)

    # Determine if multi-node (using Ray)
    backend = "ray" if (exp['pp'] > 1) else None
    gpu_monitor = None  # Will be initialized after LLM creation

    try:
        # if backend == "ray":
        #     restart_ray_cluster()

        # # Enable LMCache for KV cache management
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            kv_buffer_device="cpu"
        )
        # ktc = KVTransferConfig(
        #     kv_connector="OffloadingConnector",
        #     kv_role="kv_both",
        #     kv_connector_extra_config={{"block_size":64,"num_cpu_blocks":1000}}
        # )

        llm = LLM(
            model=exp['model'],
            tensor_parallel_size=exp['tp'],
            pipeline_parallel_size=exp['pp'],
            max_model_len=min(exp['max_input_length'] + exp['max_output_length'] -1, 32768), # -1 because of the eos token removing
            trust_remote_code=True,
            distributed_executor_backend=backend,
            gpu_memory_utilization=0.85,
            # quantization="awq",
            enforce_eager=True,
            # kv_transfer_config=ktc,
            enable_chunked_prefill=False,
            # truncate_prompt_tokens=exp['max_input_length'],
            enable_prefix_caching=False
        )

        # Initialize GPU monitor AFTER LLM creation (Ray is now initialized)
        use_distributed = backend == "ray" and ray.is_initialized()
        if use_distributed:
            print("📡 Using distributed GPU monitoring across Ray cluster")
            gpu_monitor = DistributedGPUMonitor(sample_interval=0.5)
        else:
            print("📊 Using local GPU monitoring")
            gpu_monitor = GPUMonitor(sample_interval=0.5)

        # Extract vLLM configuration info (KV cache, scheduler settings)
        vllm_config = get_vllm_config_info(llm)
        print(f"📊 vLLM Config: {{vllm_config}}")

        tokenizer = llm.get_tokenizer()
        dataset = load_dataset("emozilla/pg19-test", split="test")

        # Prepare prompts
        prompts = []
        for i in range(min(NUM_SAMPLES, len(dataset))):
            text = "Please summarize the following text: " + dataset[i]["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)[:exp['max_input_length']]
            prompts.append(tokenizer.decode(tokens))

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=exp['max_output_length'],
            min_tokens=exp['max_output_length'],
            ignore_eos=True,
        )

        # Warmup run to trigger kernel compilation and initialization
        print(f"🔥 Running warmup with {{min(5, len(prompts))}} samples...")
        _ = llm.generate(prompts[:min(5, len(prompts))], sampling_params)
        torch.cuda.synchronize()  # Wait for warmup to complete
        print(f"✅ Warmup complete, starting actual measurement")

        # Start GPU monitoring
        gpu_monitor.start()

        # Start scheduler monitoring (queue depths)
        scheduler_monitor = SchedulerMonitor(llm, sample_interval=0.25)
        scheduler_monitor.start()

        # Run and measure
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()  # Wait for all GPU work to complete
        elapsed = time.perf_counter() - start

        # Stop monitoring and get summaries
        gpu_monitor.stop()
        scheduler_monitor.stop()
        gpu_metrics = gpu_monitor.get_summary()
        scheduler_metrics = scheduler_monitor.get_summary()

        # Count tokens (vLLM native way)
        total_prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs) for o in outputs
        )

        # Generate experiment ID for timeseries file
        exp_id = f"tp{{exp['tp']}}_pp{{exp['pp']}}_in{{exp['max_input_length']}}_out{{exp['max_output_length']}}"
        timeseries_file = f"/tmp/timeseries_{{exp_id}}.json"

        # Save time-series data to separate file
        timeseries_data = {{
            'exp_id': exp_id,
            'config': exp,
            'elapsed_time': elapsed,
            'gpu_timeseries': gpu_monitor.get_timeseries(),
            'scheduler_timeseries': scheduler_monitor.get_timeseries(),
        }}
        with open(timeseries_file, 'w') as f:
            json.dump(timeseries_data, f)
        print(f"📈 Time-series saved to {{timeseries_file}}")

        time.sleep(30)
        # Cleanup BEFORE creating result
        cleanup_dist_env_and_memory(shutdown_ray=False) #DO  NOT SHUT DOWN RAY
        # Clean up lmcache backend
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            LMCacheEngineBuilder.destroy(ENGINE_NAME)
        except:
            # if backend == "ray":
            #     restart_ray_cluster()
            pass

        # Force GPU cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        send_discord_message(f"✅ Results saved to {{exp['tp']}}-{{exp['pp']}}")

        # Calculate cost efficiency metrics
        total_tokens = total_prompt_tokens + total_output_tokens
        elapsed_hours = elapsed / 3600
        cost_for_run = PRICE_PER_HOUR * elapsed_hours
        tokens_per_dollar = round(total_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        input_tokens_per_dollar = round(total_prompt_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        output_tokens_per_dollar = round(total_output_tokens / cost_for_run, 2) if cost_for_run > 0 else 0

        # Build result with all metrics (summary only, timeseries in separate file)
        result = {{
            **exp,
            'exp_id': exp_id,
            'elapsed_time': elapsed,
            'total_prompt_tokens': total_prompt_tokens,
            'total_output_tokens': total_output_tokens,
            'requests_per_sec': round(len(outputs) / elapsed, 3),
            'input_tokens_per_sec': round(total_prompt_tokens / elapsed, 2),
            'output_tokens_per_sec': round(total_output_tokens / elapsed, 2),
            'total_tokens_per_sec': round((total_prompt_tokens + total_output_tokens) / elapsed, 2),
            'status': 'success',
            'timeseries_file': timeseries_file,
            # Infrastructure info
            'instance_type': INSTANCE_TYPE,
            'price_per_hour': PRICE_PER_HOUR,
            'num_nodes': NUM_NODES,
            'gpus_per_node': GPUS_PER_NODE,
            'total_gpus': NUM_NODES * GPUS_PER_NODE,
            # Cost efficiency metrics
            'cost_for_run_usd': round(cost_for_run, 4),
            'tokens_per_dollar': tokens_per_dollar,
            'input_tokens_per_dollar': input_tokens_per_dollar,
            'output_tokens_per_dollar': output_tokens_per_dollar,
            # vLLM config info
            **{{f'config_{{k}}': v for k, v in vllm_config.items()}},
            # GPU utilization metrics (summary)
            **gpu_metrics,
            # Scheduler queue metrics (summary)
            **scheduler_metrics,
        }}
        
    except Exception as e:
        # Stop GPU monitor on error (if it was initialized)
        if gpu_monitor is not None:
            gpu_monitor.stop()

        import traceback
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            error_msg = f"OOM: {{error_msg[:200]}}"
        send_discord_message(f"❌ Cluster {{exp['tp']}}-{{exp['pp']}} FAILED: {{error_msg}}")
        # if backend == "ray":
        #     restart_ray_cluster()
        result = {{**exp, 'status': 'error', 'error': error_msg}}
        
        # Force GPU cleanup on error
        try:
            cleanup_dist_env_and_memory(shutdown_ray=False) #DO  NOT SHUT DOWN RAY
            # if 'llm' in locals():
                # del llm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass
    return result

def main():
    results = []
    for i, exp in enumerate(EXPERIMENTS):
        # if i in [0,1,2,3]:
        #     continue
        result = run_benchmark(exp)
        results.append(result)
        print(f"Result: {{result}}")
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\\n✅ All done! Results in {{RESULTS_FILE}}")

if __name__ == "__main__":
    main()
'''

def run_cluster_benchmarks(cluster_config, experiments, parent_dir=None, dry_run=True, gpu_type=DEFAULT_GPU_TYPE):
    gpus_per_node, num_nodes = cluster_config
    # Use TP/PP from the first experiment for naming (all experiments in a group have compatible TP/PP)
    tp = experiments[0]['tp']
    pp = experiments[0]['pp']
    # Include input/output length in cluster name to avoid conflicts across IO groups
    input_len = experiments[0]['max_input_length']
    output_len = experiments[0]['max_output_length']
    cluster_name = f"roofline-tp{tp}-pp{pp}-{input_len}in-{output_len}out"

    # Create consolidated result directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"tp{tp}-pp{pp}-{timestamp}"
    
    # Get instance family and GPU name for directory organization
    gpu_config = GPU_CONFIGS[gpu_type]
    instance_family = gpu_config["instance_family"]
    # Use gpu_type as GPU name (e.g., "L4", "L40S", "A100-40gb", "A100-80gb")
    gpu_name = gpu_type
    instance_dir = f"aws-{instance_family}-{gpu_name}"
    
    # If parent_dir is provided, create subdirectory structure: parent_dir/aws-{instance_family}-{gpu_name}/tp{tp}-pp{pp}-{timestamp}
    if parent_dir:
        parent_path = Path(parent_dir)
        parent_path.mkdir(exist_ok=True)
        instance_path = parent_path / instance_dir
        instance_path.mkdir(exist_ok=True)
        result_dir = instance_path / subdir_name
    else:
        # If no parent_dir, create: aws-{instance_family}-{gpu_name}/tp{tp}-pp{pp}-{timestamp}
        instance_path = Path(instance_dir)
        instance_path.mkdir(exist_ok=True)
        result_dir = instance_path / subdir_name

    print(f"\n{'='*70}")
    print(f"Cluster: {cluster_name}")
    print(f"Config: {gpus_per_node} GPUs/node × {num_nodes} nodes")
    print(f"Experiments: {len(experiments)}")
    print("="*70)

    if dry_run:
        print("[DRY RUN] Would launch and run:")
        for exp in experiments:
            print(f"  - TP={exp['tp']}, PP={exp['pp']}, "
                  f"in={exp['max_input_length']}, out={exp['max_output_length']}")
        if len(experiments) > 3:
            print(f"  ... and {len(experiments)-3} more")
        return []

    # Create result directory and set up logging
    result_dir.mkdir(exist_ok=True)
    log_file = result_dir / "benchmark.log"
    setup_logger(log_file)

    logger.info(f"Results will be saved to: {result_dir}")

    # Define paths within result directory
    local_results = result_dir / "results.json"

    # 1. Write YAML
    work_dir = Path("roofline_benchmarks")
    work_dir.mkdir(exist_ok=True)
    yaml_path = work_dir / f"{cluster_name}.yaml"
    script_path = work_dir / f"benchmark_{cluster_name}.py"
    yaml_content = generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type)
    yaml_path.write_text(yaml_content)

    # 2. Write benchmark script
    script_content = generate_benchmark_script(experiments, gpus_per_node, num_nodes, gpu_type)
    script_path.write_text(script_content)

    # Track active cluster for cleanup on unexpected exit
    set_active_cluster(cluster_name)

    try:
        # 3. Launch cluster and capture output
        logger.info(f"🚀 Launching {cluster_name}...")
        logger.info(f"YAML to run: {yaml_path}")
        logger.info(f"Running on cluster: {cluster_name}")

        # Run sky launch and capture output (with retry-until-up to handle capacity issues)
        process = subprocess.Popen(
            ["sky", "launch", "-y", "--retry-until-up", "-c", cluster_name, str(yaml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output to both console and log file
        for line in process.stdout:
            line = line.rstrip()
            logger.info(line)

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "sky launch")

        # 5. Fetch results
        logger.info(f"📥 Fetching results...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/benchmark_results.json",
            str(local_results)
        ], check=True)

        # Fetch timeseries files directly into result directory
        logger.info(f"📈 Fetching timeseries data...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/timeseries_*.json",
            str(result_dir)  # Path object converts to string correctly
        ], check=False)  # Don't fail if no timeseries files

        # Update results with local timeseries paths
        with open(local_results) as f:
            results = json.load(f)

        for result in results:
            if 'timeseries_file' in result and result.get('status') == 'success':
                # Update path to local location
                remote_filename = Path(result['timeseries_file']).name
                local_ts_path = result_dir / remote_filename
                result['timeseries_file'] = str(local_ts_path)

        # Save updated results
        with open(local_results, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Results saved to {result_dir}/")
        send_discord_message(f"✅ Results saved to {result_dir}")
        return results

    except Exception as e:
        # Mark ALL experiments in this cluster as failed
        logger.error(f"❌ Cluster error: {e}")
        logger.info(f"📥 Attempting to fetch partial results...")
        send_discord_message(f"❌ Cluster {cluster_name} FAILED: {str(e)[:100]}")
        try:
            logger.info(f"📥 Fetching results...")
            subprocess.run([
                "scp",
                f"{cluster_name}:/tmp/benchmark_results.json",
                str(local_results)
            ], check=True)
        except:
            pass
        failed_results = [{**exp, 'status': 'cluster_error', 'error': str(e)} for exp in experiments]
        with open(local_results, 'w') as f:
            json.dump(failed_results, f, indent=2)
        return failed_results

    finally:
        # 6. Teardown
        logger.info(f"🗑️  Tearing down {cluster_name}...")
        subprocess.run(["sky", "down", "-y", cluster_name])

        # Clear active cluster after successful teardown
        set_active_cluster(None)




def save_results_csv(all_results, output_path="benchmark_results.csv"):
    """Save all results to CSV."""
    if not all_results:
        return
    # Collect all unique fieldnames from all results
    fieldnames = []
    for result in all_results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"📊 Saved {len(all_results)} results to {output_path}")


def main():
    # Check for orphaned clusters from previous crashed runs
    check_orphaned_cluster()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run benchmark experiments on AWS GPU instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Available GPU types: {', '.join(GPU_CONFIGS.keys())}
Default: {DEFAULT_GPU_TYPE}

Examples:
  python automatic_launch_1.py --gpu L40S
  python automatic_launch_1.py experiment.csv --gpu A100 --run
  python automatic_launch_1.py --gpu H100 --run
        '''
    )
    parser.add_argument('csv_file', nargs='?', default='experiment_L40_llama.csv',
                       help='CSV file with experiment configurations (default: experiment_L40_llama.csv)')
    parser.add_argument('--gpu', '--gpu-type', dest='gpu_type', 
                       choices=list(GPU_CONFIGS.keys()),
                       default=DEFAULT_GPU_TYPE,
                       help=f'GPU type to use (default: {DEFAULT_GPU_TYPE})')
    parser.add_argument('--run', action='store_true',
                       help='Actually launch clusters (default: dry run)')
    
    args = parser.parse_args()
    
    csv_path = args.csv_file
    gpu_type = args.gpu_type
    dry_run = not args.run

    print(f"🔧 Using GPU type: {gpu_type}")
    print(f"📊 Loading experiments from: {csv_path}")
    if dry_run:
        print("💡 DRY RUN mode - add --run to actually launch clusters")

    experiments = load_experiments(csv_path)
    io_groups = group_by_input_output_then_cluster(experiments, gpu_type)

    print(f"📊 Loaded {len(experiments)} experiments")
    print(f"📦 Grouped by input/output length, then by cluster config:")
    for (input_len, output_len), cluster_groups in sorted(io_groups.items()):
        print(f"   {input_len}in/{output_len}out: {len(cluster_groups)} cluster configs")
        for (gpus, nodes), exps in sorted(cluster_groups.items()):
            print(f"      {gpus} GPU/node × {nodes} nodes: {len(exps)} experiments")

    if dry_run:
        print("\n💡 This is a DRY RUN. Add --run to actually launch clusters.")

    # Run each group and collect all results
    all_results = []

    def cluster_sort_key(item):
        (gpus_per_node, num_nodes), _ = item
        # Cost/spot-friendly: TP asc (1,4,8), then PP asc (1..4)
        return (gpus_per_node, num_nodes)

    try:
        for (input_len, output_len), cluster_groups in sorted(io_groups.items()):
            # Create parent directory for this input/output length
            parent_dir = f"result-{input_len}in_{output_len}out"
            
            for cluster_config, exps in sorted(cluster_groups.items(), key=cluster_sort_key):
                results = run_cluster_benchmarks(cluster_config, exps, parent_dir=parent_dir, dry_run=dry_run, gpu_type=gpu_type)
                if results:
                    all_results.extend(results)

        # Save combined results to CSV with datetime
        if all_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_output = f"benchmark_results-{timestamp}.csv"
            save_results_csv(all_results, csv_output)
    except Exception as e:
        print(f"\n❌ Unexpected error in main: {e}")
        print("   Cleanup will be triggered automatically...")
        raise  # Re-raise to trigger atexit handler


if __name__ == "__main__":
    main()


    # some notes
    # without lmcache, vllm v0.10.2 worked for tp4,pp3 (almost?)
    # now testing with vllm v0.11.0 and no lmcache (this also worked)
    # have to test with vllm v0.11.0 and lmcache v0.3.9 (trying this now) (didnt work, one exp worked, rest didnt, mostly due to ray dag)
    # have to test with vllm v0.11.0 and lmcache v0.3.10 (same issue)
    # have to test with vllm v0.13.0 and lmcache v0.3.10 (same issue) (this issue is interesting - layers not found? wtf?)
    # there is something called as a lmcache mp connector, will check into that as well
    # need to check if ray restarts help withn vllmv0.11 and v0.3.9 (does not help)
    # my suspicion is that i had not put the time between two llm establishments and cleanup_dist_env_and_memory was not used (this HELPS!)
    # export VLLM_USE_RAY_WRAPPED_PP_COMM=0 maybe trying this helps? (same as all)
    # have to test with vllm v0.13.0 and lmcache v0.3.11 (latest versions, and building from source????)
    # this seems like another flag to try enable_async_loading: True (didnt do anything significiant)
    # vllm with v0.10.2 is NOT WORKING for pp=4 tp=4, maybe thats also an issue, 
    # will try offloading connector as well

    # TP=4/PP=4 worked with vllm v0.10.0 and max_gpu_util = 0.85 and 
    # export NCCL_P2P_DISABLE=1
    # export NCCL_TIMEOUT=1800
    # export TORCH_NCCL_BLOCKING_WAIT=1
    # export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
    # export TORCH_DISTRIBUTED_DEBUG=DETAIL  
    # export NCCL_DEBUG=INFO 

    # trying lmcache w it (didnt work tbh)
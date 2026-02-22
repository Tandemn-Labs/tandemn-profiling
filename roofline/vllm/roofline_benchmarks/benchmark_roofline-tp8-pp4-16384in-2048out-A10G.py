#!/usr/bin/env python3
import os
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
os.environ["LMCACHE_LOG_LEVEL"] = "INFO"

import requests
import json
import time
import torch
import gc

# Verify CUDA is available before importing vLLM
# Note: Fabric Manager must be running for multi-GPU A100 systems (handled in setup_commands)
# We do minimal CUDA checking here to avoid creating tensors that conflict with vLLM's InferenceMode
print("🔧 Verifying CUDA availability before vLLM import...")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print(f"   ✅ CUDA available: {device_count} GPU(s) detected")
    else:
        raise RuntimeError("CUDA reports available but no devices found")
else:
    raise RuntimeError("CUDA is not available! Check if Fabric Manager is running for A100 systems.")

import ray
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Handle API differences between vLLM versions
try:
    from vllm.config import KVTransferConfig
except ImportError:
    KVTransferConfig = None

try:
    from vllm.distributed import cleanup_dist_env_and_memory
except ImportError:
    # Fallback for older vLLM versions
    def cleanup_dist_env_and_memory(shutdown_ray=True):
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        if shutdown_ray:
            ray.shutdown()

import shutil
import subprocess
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
        self._device_count = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
            self._device_count = pynvml.nvmlDeviceGetCount()
            print(f"[{self.node_id}] GPU monitor initialized with {self._device_count} GPUs")
        except Exception as e:
            print(f"[{self.node_id}] pynvml not available: {e}")
            import traceback
            traceback.print_exc()
            self._device_count = 0

    def start(self):
        """Start collecting samples."""
        if not self._pynvml_available:
            print(f"[{self.node_id}] Cannot start monitoring: pynvml not available")
            return
        try:
            self.timeseries = []
            self._start_time = time.time()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            print(f"[{self.node_id}] Monitoring thread started")
        except Exception as e:
            print(f"[{self.node_id}] Failed to start monitoring thread: {e}")
            import traceback
            traceback.print_exc()

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
            sample = {'t': round(relative_time, 3)}

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
                    sample[f'{self.node_id}_gpu{i}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'{self.node_id}_gpu{i}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'{self.node_id}_gpu{i}_sm_pct'] = gpu_util_pct
                    sample[f'{self.node_id}_gpu{i}_membw_pct'] = mem_bw_util_pct
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
    
    def get_ray_node_id(self):
        """Return the Ray node ID where this actor is running."""
        try:
            import ray
            # Get the current node's resource ID
            node_id = ray.get_runtime_context().get_node_id()
            return node_id
        except:
            return None

    def get_device_count(self):
        """Return number of GPUs on this node."""
        return self._device_count
    
    def health_check(self):
        """Simple health check to verify actor is responsive."""
        return {"status": "ok", "node_id": self.node_id, "gpus": self._device_count}


class DistributedGPUMonitor:
    """Manage GPU monitor actors across all Ray nodes."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.actors = []
        self.node_map = {}  # node_id -> actor

    def start(self):
        """Deploy actors on all Ray nodes and start monitoring."""
        # Get all Ray nodes
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n.get('Alive', False)]

        print(f"📡 Found {len(alive_nodes)} Ray nodes for GPU monitoring")

        # Try using placement group to ensure one actor per node
        # If that fails, fall back to creating actors without pinning
        try:
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            
            # Create a placement group with one bundle per node
            bundles = [{"CPU": 0.1} for _ in alive_nodes]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready(), timeout=60.0)  # Increased timeout for heavy Ray load
            print(f"   Created placement group with {len(bundles)} bundles")
            
            # Create actors using placement group
            for idx, node in enumerate(alive_nodes):
                node_id = f"node{idx}"
                node_ip = node.get('NodeManagerAddress', 'unknown')
                
                try:
                    actor = GPUMonitorActor.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=idx
                        ),
                        num_cpus=0.1,
                    ).remote(node_id, self.sample_interval)
                    
                    self.actors.append(actor)
                    self.node_map[node_id] = actor
                    print(f"  ✓ Deployed monitor on {node_id} ({node_ip}) using placement group")
                except Exception as e:
                    print(f"  ✗ Failed to deploy on {node_id}: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as pg_error:
            print(f"   Placement group approach failed: {pg_error}")
            print(f"   Falling back to simple actor creation (no pinning)...")
            
            # Fallback: create actors without pinning, let Ray schedule them
            for idx, node in enumerate(alive_nodes):
                node_id = f"node{idx}"
                node_ip = node.get('NodeManagerAddress', 'unknown')
                
                try:
                    actor = GPUMonitorActor.options(
                        num_cpus=0.1,
                    ).remote(node_id, self.sample_interval)
                    
                    self.actors.append(actor)
                    self.node_map[node_id] = actor
                    print(f"  ✓ Deployed monitor actor (will query node location)")
                    
                    # Try to get the Ray node where the actor is actually running
                    try:
                        ray_node_id = ray.get(actor.get_ray_node_id.remote(), timeout=15.0)  # Increased timeout
                        print(f"     Actor running on Ray node: {ray_node_id}")
                    except Exception as node_check_err:
                        print(f"     Could not determine actor node: {node_check_err}")
                except Exception as e:
                    print(f"  ✗ Failed to deploy on {node_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Start all actors with individual timeouts to avoid hanging
        # Use longer timeouts when Ray is under heavy load (e.g., from vLLM)
        print(f"   Starting monitoring on {len(self.actors)} nodes...")
        started_count = 0
        for idx, actor in enumerate(self.actors):
            try:
                # Start each actor individually with timeout
                # Increased timeout to handle Ray being busy with vLLM operations
                print(f"   Starting monitor on node{idx}...")
                future = actor.start.remote()
                ray.get(future, timeout=30.0)  # Increased from 10s to 30s for heavy Ray load
                started_count += 1
                print(f"   ✓ Started monitor on node{idx}")
            except Exception as e:
                print(f"   ✗ Failed to start monitor on node{idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other actors
        
        if started_count > 0:
            print(f"📊 GPU monitoring started on {started_count}/{len(self.actors)} nodes")
            return True  # Success
        else:
            print(f"⚠️  Warning: No actors started successfully. Falling back to local monitoring.")
            # Clear actors so fallback works
            self.actors = []
            self.node_map = {}  # Empty dict - escaped for f-string
            return False  # Failed - should fall back to local

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
        merged = {}
        for series in all_series:
            for sample in series:
                t = sample['t']
                if t not in merged:
                    merged[t] = {'t': t}
                for key, value in sample.items():
                    if key != 't':
                        merged[t][key] = value

        # Sort by timestamp and return as list
        return [merged[t] for t in sorted(merged.keys())]

    def get_summary(self):
        """Return summary statistics across all nodes."""
        timeseries = self.get_timeseries()
        if not timeseries:
            return {}

        summary = {}
        metrics = defaultdict(list)

        for sample in timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        # Per-GPU summaries
        for key, values in metrics.items():
            if values:
                summary[f'{key}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{key}_max'] = round(max(values), 2)
                summary[f'{key}_min'] = round(min(values), 2)

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
        self.timeseries = []  # List of {timestamp, gpu0_*, gpu1_*, ...}
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
            print(f"⚠️  pynvml not available, GPU monitoring disabled: {e}")

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
            sample = {'t': round(relative_time, 3)}

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

                    sample[f'gpu{i}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'gpu{i}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'gpu{i}_sm_pct'] = gpu_util_pct
                    sample[f'gpu{i}_membw_pct'] = mem_bw_util_pct

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
            return {}

        summary = {}

        # Collect values per metric
        metrics = defaultdict(list)
        for sample in self.timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        # Per-GPU summaries
        for key, values in metrics.items():
            if values:
                summary[f'{key}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{key}_max'] = round(max(values), 2)
                summary[f'{key}_min'] = round(min(values), 2)

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
                sample = {'t': round(relative_time, 3)}

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
                    print(f"⚠️  SchedulerMonitor error: {e}")

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return raw time-series data for plotting."""
        return self.timeseries

    def get_summary(self):
        """Return summary statistics of scheduler queue depths."""
        if not self.timeseries:
            return {}

        summary = {}

        # Collect all unique keys
        all_keys = set()
        for sample in self.timeseries:
            all_keys.update(sample.keys())
        all_keys.discard('t')

        for key in all_keys:
            values = [s.get(key, 0) for s in self.timeseries if key in s]
            if values:
                summary[f'{key}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{key}_max'] = max(values)

        summary['scheduler_samples'] = len(self.timeseries)
        return summary

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453154642706960485/iFXIAaDTLxNO7_GHKHhXnXwFFnXziniP4TUwLUDUnXHtT9kNo08eQBjGQ4CiBr6AazY6"

def send_discord_message(message):
    payload = {"content": message}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)

# Clean up existing cache directory
cache_dir = "/tmp/lmcache_disk"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

os.makedirs(cache_dir, exist_ok=True)

EXPERIMENTS = [
  {
    "tp": 8,
    "pp": 4,
    "max_input_length": 16384,
    "max_output_length": 2048,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  }
]
RESULTS_FILE = "/tmp/benchmark_results.json"
NUM_SAMPLES = 30

# Cluster pricing information
INSTANCE_TYPE = "4x g5.48xlarge"
PRICE_PER_HOUR = 65.536
NUM_NODES = 4
GPUS_PER_NODE = 8


def get_vllm_config_info(llm):
    """Extract all configuration info from vLLM engine."""
    config_info = {}
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
            config_info['swap_space'] = getattr(cache_cfg, 'swap_space', None)
            config_info['cache_dtype'] = getattr(cache_cfg, 'cache_dtype', None)
            config_info['kv_cache_dtype'] = getattr(cache_cfg, 'kv_cache_dtype', None)

        # Scheduler config (concurrency settings)
        if hasattr(engine, 'scheduler_config'):
            sched_cfg = engine.scheduler_config
            # Extract scheduler config attributes
            config_info['max_num_seqs'] = getattr(sched_cfg, 'max_num_seqs', None)
            config_info['max_num_batched_tokens'] = getattr(sched_cfg, 'max_num_batched_tokens', None)
            config_info['max_model_len'] = getattr(sched_cfg, 'max_model_len', None)
            config_info['chunked_prefill_enabled'] = getattr(sched_cfg, 'chunked_prefill_enabled', None)
            config_info['max_paddings'] = getattr(sched_cfg, 'max_paddings', None)
            config_info['delay_factor'] = getattr(sched_cfg, 'delay_factor', None)
            config_info['chunked_prefill_size'] = getattr(sched_cfg, 'chunked_prefill_size', None)
            config_info['max_seq_len'] = getattr(sched_cfg, 'max_seq_len', None)
            config_info['policy'] = getattr(sched_cfg, 'policy', None)
            config_info['long_prefill_token_threshold'] = getattr(sched_cfg, 'long_prefill_token_threshold', None)
            config_info['max_long_partial_prefills'] = getattr(sched_cfg, 'max_long_partial_prefills', None)
        
        # Also check scheduler instance (not just config) for runtime values
        if hasattr(engine, 'scheduler'):
            scheduler = engine.scheduler
            # Try to get runtime scheduler values if config doesn't have them
            if config_info.get('max_num_seqs') is None:
                config_info['max_num_seqs'] = getattr(scheduler, 'max_num_seqs', None)
            if config_info.get('max_num_batched_tokens') is None:
                config_info['max_num_batched_tokens'] = getattr(scheduler, 'max_num_batched_tokens', None)
        
        # Ensure these keys exist even if scheduler_config doesn't exist or attributes are missing
        if 'max_num_seqs' not in config_info:
            config_info['max_num_seqs'] = None
        if 'max_num_batched_tokens' not in config_info:
            config_info['max_num_batched_tokens'] = None

        # Model config
        if hasattr(engine, 'model_config'):
            model_cfg = engine.model_config
            config_info['dtype'] = str(getattr(model_cfg, 'dtype', None))
            config_info['max_model_len'] = getattr(model_cfg, 'max_model_len', None)
            config_info['quantization'] = getattr(model_cfg, 'quantization', None)
            config_info['quantization_param_path'] = getattr(model_cfg, 'quantization_param_path', None)
            config_info['enforce_eager'] = getattr(model_cfg, 'enforce_eager', None)
            config_info['max_seq_len_to_capture'] = getattr(model_cfg, 'max_seq_len_to_capture', None)

        # Parallel config
        if hasattr(engine, 'parallel_config'):
            parallel_cfg = engine.parallel_config
            config_info['tensor_parallel_size'] = getattr(parallel_cfg, 'tensor_parallel_size', None)
            config_info['pipeline_parallel_size'] = getattr(parallel_cfg, 'pipeline_parallel_size', None)
            config_info['world_size'] = getattr(parallel_cfg, 'world_size', None)
            config_info['rank'] = getattr(parallel_cfg, 'rank', None)

        # Decoding config
        if hasattr(engine, 'decoding_config'):
            decoding_cfg = engine.decoding_config
            config_info['use_chunked_prefill'] = getattr(decoding_cfg, 'use_chunked_prefill', None)
            # Prefer decoding_config value, but keep scheduler_config value if decoding_config doesn't have it
            decoding_max_batched = getattr(decoding_cfg, 'max_num_batched_tokens', None)
            if decoding_max_batched is not None:
                config_info['max_num_batched_tokens'] = decoding_max_batched
            # If not set yet, ensure it's at least None
            if 'max_num_batched_tokens' not in config_info:
                config_info['max_num_batched_tokens'] = None

        # Check for LMCache configuration
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            if hasattr(LMCacheEngineBuilder, 'get_engine'):
                lmcache_engine = LMCacheEngineBuilder.get_engine(ENGINE_NAME)
                if lmcache_engine:
                    config_info['lmcache_enabled'] = True
                    # Try to get LMCache config
                    if hasattr(lmcache_engine, 'config'):
                        lmcache_cfg = lmcache_engine.config
                        config_info['lmcache_chunk_size'] = getattr(lmcache_cfg, 'chunk_size', None)
                        config_info['lmcache_local_cpu'] = getattr(lmcache_cfg, 'local_cpu', None)
                        config_info['lmcache_max_local_cpu_size'] = getattr(lmcache_cfg, 'max_local_cpu_size', None)
                        config_info['lmcache_use_layerwise'] = getattr(lmcache_cfg, 'use_layerwise', None)
                        config_info['lmcache_enable_lazy_memory_allocator'] = getattr(lmcache_cfg, 'enable_lazy_memory_allocator', None)
                    # Check environment variables
                    import os
                    config_info['lmcache_use_experimental'] = os.environ.get('LMCACHE_USE_EXPERIMENTAL', None)
                    config_info['lmcache_enable_async_loading'] = os.environ.get('LMCACHE_ENABLE_ASYNC_LOADING', None)
                    config_info['lmcache_remote_serde'] = os.environ.get('LMCACHE_REMOTE_SERDE', None)
                else:
                    config_info['lmcache_enabled'] = False
            else:
                config_info['lmcache_enabled'] = False
        except Exception:
            config_info['lmcache_enabled'] = False

        # Check for KV transfer config (LMCache connector)
        if hasattr(llm, 'llm_engine') and hasattr(engine, 'cache_config'):
            cache_cfg = engine.cache_config
            if hasattr(cache_cfg, 'kv_transfer_config') and cache_cfg.kv_transfer_config:
                ktc = cache_cfg.kv_transfer_config
                config_info['kv_transfer_config_enabled'] = True
                config_info['kv_connector'] = getattr(ktc, 'kv_connector', None)
                config_info['kv_role'] = getattr(ktc, 'kv_role', None)
                config_info['kv_buffer_device'] = getattr(ktc, 'kv_buffer_device', None)
                if hasattr(ktc, 'kv_connector_extra_config'):
                    config_info['kv_connector_extra_config'] = ktc.kv_connector_extra_config
            else:
                config_info['kv_transfer_config_enabled'] = False

        # LLM initialization parameters (from the LLM object)
        config_info['enable_prefix_caching'] = getattr(llm, 'enable_prefix_caching', None)
        config_info['enable_chunked_prefill'] = getattr(llm, 'enable_chunked_prefill', None)
        config_info['trust_remote_code'] = getattr(llm, 'trust_remote_code', None)

    except Exception as e:
        print(f"⚠️  Could not extract vLLM config: {e}")
        import traceback
        traceback.print_exc()

    return config_info

def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\n{'='*70}")
    print(f"Running: TP={exp['tp']}, PP={exp['pp']}, "
          f"input={exp['max_input_length']}, output={exp['max_output_length']}")
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
        #     kv_connector_extra_config={"block_size":64,"num_cpu_blocks":1000}
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
        # For PP > 1, we should use distributed monitoring. vLLM initializes Ray internally,
        # but we need to explicitly connect to it using RAY_ADDRESS.
        use_distributed = False
        if backend == "ray":
            print("📡 Attempting distributed GPU monitoring across Ray cluster...")
            try:
                # vLLM initializes Ray internally, but we need to connect to it explicitly
                # Get Ray address from environment (set at the top of the script)
                ray_address = os.environ.get("RAY_ADDRESS", "127.0.0.1:6379")
                
                # Try to connect to Ray cluster explicitly
                ray_available = False
                try:
                    # If Ray is not initialized, try to connect to the existing cluster
                    if not ray.is_initialized():
                        print(f"   Connecting to Ray cluster at {ray_address}...")
                        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
                        print(f"   ✅ Connected to Ray cluster")
                    
                    # Now check if we can access nodes
                    nodes = ray.nodes()
                    alive_nodes = [n for n in nodes if n.get('Alive', False)]
                    ray_available = len(alive_nodes) > 0
                    print(f"   Ray cluster detected: {len(alive_nodes)} alive nodes")
                    for idx, node in enumerate(alive_nodes):
                        node_id = node.get('NodeID', 'unknown')
                        print(f"      Node {idx}: {node_id}")
                except Exception as ray_check_error:
                    print(f"   Could not access Ray cluster: {ray_check_error}")
                    print(f"   Ray.is_initialized(): {ray.is_initialized() if hasattr(ray, 'is_initialized') else 'N/A'}")
                    print(f"   RAY_ADDRESS: {ray_address}")
                
                if ray_available:
                    # Try to initialize distributed monitor
                    gpu_monitor = DistributedGPUMonitor(sample_interval=0.5)
                    use_distributed = True
                    print("✅ Distributed GPU monitoring initialized successfully")
                else:
                    raise Exception("Ray cluster not accessible")
            except Exception as e:
                print(f"⚠️  Failed to initialize distributed GPU monitor: {e}")
                import traceback
                traceback.print_exc()
                print("📊 Falling back to local GPU monitoring")
                gpu_monitor = GPUMonitor(sample_interval=0.5)
                use_distributed = False
        else:
            print("📊 Using local GPU monitoring (single node, no Ray)")
            gpu_monitor = GPUMonitor(sample_interval=0.5)

        # Extract vLLM configuration info (KV cache, scheduler settings)
        vllm_config = get_vllm_config_info(llm)
        print(f"📊 vLLM Config: {vllm_config}")

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
        print(f"🔥 Running warmup with {min(5, len(prompts))} samples...")
        _ = llm.generate(prompts[:min(5, len(prompts))], sampling_params)
        torch.cuda.synchronize()  # Wait for warmup to complete
        print(f"✅ Warmup complete, starting actual measurement")

        # Start GPU monitoring
        # If distributed monitoring fails, it will fall back gracefully
        if use_distributed:
            try:
                success = gpu_monitor.start()
                if not success or (hasattr(gpu_monitor, 'actors') and len(gpu_monitor.actors) == 0):
                    print("⚠️  Distributed monitoring failed, falling back to local")
                    raise Exception("Distributed monitoring not available")
            except Exception as monitor_error:
                print(f"⚠️  Failed to start distributed GPU monitoring: {monitor_error}")
                print("📊 Falling back to local GPU monitoring")
                gpu_monitor = GPUMonitor(sample_interval=0.5)
                use_distributed = False
                gpu_monitor.start()
        else:
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
        
        # Log monitoring info for debugging
        if use_distributed and hasattr(gpu_monitor, 'actors'):
            print(f"📊 GPU monitoring: {len(gpu_monitor.actors)} nodes monitored")
            if 'num_nodes_monitored' in gpu_metrics:
                print(f"📊 GPU monitoring: {gpu_metrics['num_nodes_monitored']} nodes reported in summary")
        else:
            print(f"📊 GPU monitoring: local (single node)")

        # Count tokens (vLLM native way)
        total_prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs) for o in outputs
        )

        # Generate experiment ID for timeseries file
        exp_id = f"tp{exp['tp']}_pp{exp['pp']}_in{exp['max_input_length']}_out{exp['max_output_length']}"
        timeseries_file = f"/tmp/timeseries_{exp_id}.json"

        # Save time-series data to separate file
        timeseries_data = {
            'exp_id': exp_id,
            'config': exp,
            'elapsed_time': elapsed,
            'gpu_timeseries': gpu_monitor.get_timeseries(),
            'scheduler_timeseries': scheduler_monitor.get_timeseries(),
        }
        with open(timeseries_file, 'w') as f:
            json.dump(timeseries_data, f)
        print(f"📈 Time-series saved to {timeseries_file}")

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
        send_discord_message(f"✅ Results saved to {exp['tp']}-{exp['pp']}")

        # Calculate cost efficiency metrics
        total_tokens = total_prompt_tokens + total_output_tokens
        elapsed_hours = elapsed / 3600
        cost_for_run = PRICE_PER_HOUR * elapsed_hours
        tokens_per_dollar = round(total_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        input_tokens_per_dollar = round(total_prompt_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        output_tokens_per_dollar = round(total_output_tokens / cost_for_run, 2) if cost_for_run > 0 else 0

        # Collect all configuration
        benchmark_config = {
            # LLM initialization parameters
            'llm_model': exp['model'],
            'llm_tensor_parallel_size': exp['tp'],
            'llm_pipeline_parallel_size': exp['pp'],
            'llm_max_model_len': min(exp['max_input_length'] + exp['max_output_length'] - 1, 32768),
            'llm_trust_remote_code': True,
            'llm_distributed_executor_backend': backend,
            'llm_gpu_memory_utilization': 0.85,
            'llm_enforce_eager': True,
            'llm_enable_chunked_prefill': False,
            'llm_enable_prefix_caching': False,
            'llm_kv_transfer_config': None,  # Currently commented out
            # Sampling parameters
            'sampling_temperature': 0.8,
            'sampling_max_tokens': exp['max_output_length'],
            'sampling_min_tokens': exp['max_output_length'],
            'sampling_ignore_eos': True,
            # Benchmark configuration
            'benchmark_num_samples': NUM_SAMPLES,
            'benchmark_dataset': 'emozilla/pg19-test',
            'benchmark_prompt_prefix': 'Please summarize the following text: ',
            'benchmark_warmup_samples': min(5, NUM_SAMPLES),
            # GPU monitoring configuration
            'gpu_monitor_sample_interval': 0.5,
            'gpu_monitor_type': 'distributed' if use_distributed else 'local',
            # Scheduler monitoring configuration
            'scheduler_monitor_sample_interval': 0.25,
        }

        # Build result with all metrics (summary only, timeseries in separate file)
        result = {
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
            # Benchmark configuration
            **benchmark_config,
            # GPU monitoring details (added after metrics are collected)
            'gpu_monitor_num_nodes': len(gpu_monitor.actors) if (use_distributed and hasattr(gpu_monitor, 'actors')) else 1,
            'gpu_monitor_num_gpus_monitored': len([k for k in gpu_metrics.keys() if '_sm_pct_avg' in k or (k.endswith('_sm_pct_avg') and not k.startswith('avg_'))]),
            'gpu_monitor_num_nodes_reported': gpu_metrics.get('num_nodes_monitored', 1),
            # vLLM config info (extracted from engine)
            **{f'config_{k}': v for k, v in vllm_config.items()},
            # GPU utilization metrics (summary)
            **gpu_metrics,
            # Scheduler queue metrics (summary)
            **scheduler_metrics,
        }
        
    except Exception as e:
        # Stop GPU monitor on error (if it was initialized)
        if gpu_monitor is not None:
            gpu_monitor.stop()

        import traceback
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            error_msg = f"OOM: {error_msg[:200]}"
        send_discord_message(f"❌ Cluster {exp['tp']}-{exp['pp']} FAILED: {error_msg}")
        # if backend == "ray":
        #     restart_ray_cluster()
        
        # Collect configuration even for error case
        benchmark_config = {
            # LLM initialization parameters
            'llm_model': exp['model'],
            'llm_tensor_parallel_size': exp['tp'],
            'llm_pipeline_parallel_size': exp['pp'],
            'llm_max_model_len': min(exp['max_input_length'] + exp['max_output_length'] - 1, 32768),
            'llm_trust_remote_code': True,
            'llm_distributed_executor_backend': backend,
            'llm_gpu_memory_utilization': 0.85,
            'llm_enforce_eager': True,
            'llm_enable_chunked_prefill': False,
            'llm_enable_prefix_caching': False,
            'llm_kv_transfer_config': None,  # Currently commented out
            # Sampling parameters
            'sampling_temperature': 0.8,
            'sampling_max_tokens': exp['max_output_length'],
            'sampling_min_tokens': exp['max_output_length'],
            'sampling_ignore_eos': True,
            # Benchmark configuration
            'benchmark_num_samples': NUM_SAMPLES,
            'benchmark_dataset': 'emozilla/pg19-test',
            'benchmark_prompt_prefix': 'Please summarize the following text: ',
            'benchmark_warmup_samples': min(5, NUM_SAMPLES),
            # GPU monitoring configuration
            'gpu_monitor_sample_interval': 0.5,
            'gpu_monitor_type': 'distributed' if backend == "ray" else 'local',
            # Scheduler monitoring configuration
            'scheduler_monitor_sample_interval': 0.25,
        }
        
        result = {
            **exp,
            'status': 'error',
            'error': error_msg,
            # Infrastructure info
            'instance_type': INSTANCE_TYPE,
            'price_per_hour': PRICE_PER_HOUR,
            'num_nodes': NUM_NODES,
            'gpus_per_node': GPUS_PER_NODE,
            'total_gpus': NUM_NODES * GPUS_PER_NODE,
            # Benchmark configuration
            **benchmark_config,
        }
        
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
        print(f"Result: {result}")
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n✅ All done! Results in {RESULTS_FILE}")

if __name__ == "__main__":
    main()

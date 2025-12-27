import csv
import json
from dataclasses import dataclass
import subprocess
from collections import defaultdict
import sys
from pathlib import Path
import requests 

VALID_GPUS_PER_NODE = [1, 4, 8]

def get_cluster_config(tp, pp):
    if tp not in VALID_GPUS_PER_NODE:
        raise ValueError(f"Unsupported TP={tp}. Allowed: {VALID_GPUS_PER_NODE}")
    if pp < 1:
        raise ValueError(f"Invalid PP={pp}. Must be >= 1")
    return tp, pp


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

def group_by_cluster(experiments):
    groups = defaultdict(list)
    for exp in experiments:
        gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'])
        key = (gpus_per_node, num_nodes)
        groups[key].append(exp)
    return dict(groups)

def generate_yaml(gpus_per_node, num_nodes, cluster_name):
    lmcache_exports = """
  # LMCache environment variables (must be set before Ray workers start)
  export LMCACHE_USE_EXPERIMENTAL="True"
  export LMCACHE_CHUNK_SIZE="256"
  export LMCACHE_LOCAL_CPU="True"
  export LMCACHE_MAX_LOCAL_CPU_SIZE="40.0"
  export LMCACHE_SAVE_UNFULL_CHUNK="True"
  # export VLLM_USE_RAY_WRAPPED_PP_COMM=0
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
    
    ray_run = f"""{lmcache_exports}
  python roofline_benchmarks/benchmark_{cluster_name}.py
    """
    if num_nodes > 1:
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
    return f"""
name: {cluster_name}
resources:
  cloud: aws
  accelerators: L40S:{gpus_per_node}
  use_spot: false
  disk_size: 500GB
  memory: "64GB+"
  region: us-east-1
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

  uv pip install "datasets" "requests"
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

def generate_benchmark_script(experiments):
    """Generate Python benchmark script for a set of experiments."""
    exp_list = json.dumps(experiments, indent=2)
    
    return f'''#!/usr/bin/env python3
import os
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
os.environ["LMCACHE_LOG_LEVEL"] = "INFO"

import requests
import json
import time
import ray
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed import cleanup_dist_env_and_memory
import shutil
import subprocess
import time

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


def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\\n{{'='*70}}")
    print(f"Running: TP={{exp['tp']}}, PP={{exp['pp']}}, "
          f"input={{exp['max_input_length']}}, output={{exp['max_output_length']}}")
    print("="*70)
    
    try:
        backend = "ray" if (exp['pp'] > 1) else None
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
            #enable_chunked_prefill=True,
            # truncate_prompt_tokens=exp['max_input_length'],
            #enable_prefix_caching=True
        )
        
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
        
        # Run and measure
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        # Count tokens (vLLM native way)
        total_prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
        total_output_tokens = sum(
            sum(len(c.token_ids) for c in o.outputs) for o in outputs
        )
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
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        send_discord_message(f"✅ Results saved to {{exp['tp']}}-{{exp['pp']}}")
        result = {{
            **exp,
            'elapsed_time': elapsed,
            'total_prompt_tokens': total_prompt_tokens,
            'total_output_tokens': total_output_tokens,
            'requests_per_sec': len(outputs) / elapsed,
            'output_tokens_per_sec': total_output_tokens / elapsed,
            'total_tokens_per_sec': (total_prompt_tokens + total_output_tokens) / elapsed,
            'status': 'success'
        }}
        
    except Exception as e:
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
            import torch
            import gc
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

def run_cluster_benchmarks(cluster_config, experiments, dry_run=True):
    gpus_per_node, num_nodes = cluster_config
    # cluster_name = f"roofline-{gpus_per_node}gpu-{num_nodes}node"
    cluster_name = f"roofline-tp{gpus_per_node}-pp{num_nodes}"
    local_results = Path(f"results_{cluster_name}.json")
    local_logs = Path(f"logs_{cluster_name}")  # ADD THIS
    local_logs.mkdir(exist_ok=True)  # ADD THIS
    
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
    
    # 1. Write YAML
    work_dir = Path("roofline_benchmarks")
    work_dir.mkdir(exist_ok=True)
    yaml_path = work_dir / f"{cluster_name}.yaml"
    script_path = work_dir / f"benchmark_{cluster_name}.py"
    yaml_content = generate_yaml(gpus_per_node, num_nodes, cluster_name)
    # yaml_path = Path(f"/tmp/{cluster_name}.yaml")
    yaml_path.write_text(yaml_content)
    
    # 2. Write benchmark script
    script_content = generate_benchmark_script(experiments)
    # script_path = Path(f"/tmp/benchmark_{cluster_name}.py")
    script_path.write_text(script_content)
    
    try:
        # 3. Launch cluster
        print(f"🚀 Launching {cluster_name}...")
        subprocess.run(["sky", "launch", "-y", "-c", cluster_name, str(yaml_path)], check=True)
        
        # 5. Fetch results
        print(f"📥 Fetching results...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/benchmark_results.json",  
            str(local_results)
        ], check=True)
        
        print(f"✅ Results saved to {local_results}")
        send_discord_message(f"✅ Results saved to {local_results}")
        with open(local_results) as f:
            return json.load(f)
        
    except Exception as e:
        # Mark ALL experiments in this cluster as failed
        print(f"❌ Cluster error: {e}")
        print(f"📥 Attempting to fetch partial results...")
        send_discord_message(f"❌ Cluster {cluster_name} FAILED: {str(e)[:100]}")
        try:
            print(f"📥 Fetching results...")
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
        # ===== ADD LOG COLLECTION BEFORE TEARDOWN =====
        print(f"📋 Collecting logs from {cluster_name}...")
        
        # Collect logs from all nodes
        try:
            subprocess.run([
                "sky", "logs", cluster_name, 
                "--save", str(local_logs / "sky_command_output.txt")
            ], check=False)
        except Exception as log_error:
            print(f"⚠️  Warning: Could not collect all logs: {log_error}")
            send_discord_message(f"⚠️  Log collection incomplete: {str(log_error)[:100]}")
        # 6. Teardown
        print(f"🗑️  Tearing down {cluster_name}...")
        subprocess.run(["sky", "down", "-y", cluster_name])




def save_results_csv(all_results, output_path="benchmark_results.csv"):
    """Save all results to CSV."""
    if not all_results:
        return
    fieldnames = list(all_results[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"📊 Saved {len(all_results)} results to {output_path}")


def main():
    csv_path = "experiment_l40_llama.csv"
    dry_run = "--run" not in sys.argv
    
    experiments = load_experiments(csv_path)
    groups = group_by_cluster(experiments)
    
    print(f"📊 Loaded {len(experiments)} experiments")
    print(f"📦 Grouped into {len(groups)} cluster configurations:")
    for (gpus, nodes), exps in sorted(groups.items()):
        print(f"   {gpus} GPU/node × {nodes} nodes: {len(exps)} experiments")
    
    if dry_run:
        print("\n💡 This is a DRY RUN. Add --run to actually launch clusters.")
    
    # Run each group and collect all results
    all_results = []
    count = 0
    
    def cluster_sort_key(item):
        (gpus_per_node, num_nodes), _ = item
        # Cost/spot-friendly: TP asc (1,4,8), then PP asc (1..4)
        return (gpus_per_node, num_nodes)

    for cluster_config, exps in sorted(groups.items(), key=cluster_sort_key):
        if count == 10: # 6,7 failed
            results = run_cluster_benchmarks(cluster_config, exps, dry_run=dry_run)
            if results:
                all_results.extend(results)
            break
        else:
            count+=1
            continue
    
    # Save combined results to CSV
    if all_results:
        save_results_csv(all_results)


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
import csv
import json
from dataclasses import dataclass
import subprocess
from collections import defaultdict
import sys
from pathlib import Path


VALID_GPUS_PER_NODE=[8,4,1]

def get_cluster_config(tp,pp):
    world_size=tp*pp
    for gpus in VALID_GPUS_PER_NODE:
        if world_size % gpus == 0:
            return gpus, world_size // gpus
    return 1, world_size

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
    ray_run = ""
    if num_nodes > 1:
        ray_run = """
        # Start Ray cluster across all nodes
        export RAY_CMD="uv run ray"
        export RAY_DASHBOARD_HOST=0.0.0.0
        export RAY_HEAD_PORT=6379
        export RAY_DASHBOARD_PORT=8265
        export RAY_ADDRESS="127.0.0.1:6379"

        # Start Ray head + workers on all nodes
        ~/sky_templates/ray/start_cluster

        if [ "${SKYPILOT_NODE_RANK}" = "0" ]; then
            echo "Ray cluster started on head node"
            ray status --address="127.0.0.1:${RAY_HEAD_PORT}" || true
        fi
        """
    return f"""
name: {cluster_name}
resources:
  cloud: aws
  accelerators: L40S:{gpus_per_node}
  use_spot: false
num_nodes: {num_nodes}
workdir: .
setup: | 
  set -euxo pipefail
  python3 -m pip install -U pip
  python3 -m pip install -U uv

  uv venv --python 3.12 --seed
  source .venv/bin/activate

  # Install vLLM (pin as desired)
  uv pip install "vllm" "datasets"

run: |
  set -euxo pipefail
  source .venv/bin/activate {ray_run}

  echo "Cluster ready for benchmarking"
"""

def generate_benchmark_script(experiments):
    """Generate Python benchmark script for a set of experiments."""
    exp_list = json.dumps(experiments, indent=2)
    
    return f'''#!/usr/bin/env python3
import os
import json
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams

EXPERIMENTS = {exp_list}
RESULTS_FILE = "/tmp/benchmark_results.json"
NUM_SAMPLES = 100

def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\\n{{'='*70}}")
    print(f"Running: TP={{exp['tp']}}, PP={{exp['pp']}}, "
          f"input={{exp['max_input_length']}}, output={{exp['max_output_length']}}")
    print("="*70)
    
    try:
        backend = "ray" if (exp['tp'] > 1 or exp['pp'] > 1) else None

        llm = LLM(
            model=exp['model'],
            tensor_parallel_size=exp['tp'],
            pipeline_parallel_size=exp['pp'],
            max_model_len=exp['max_input_length'] + exp['max_output_length'],
            trust_remote_code=True,
            distributed_executor_backend=backend,
            gpu_memory_utilization=0.95,
            quantization="awq"
        )
        
        tokenizer = llm.get_tokenizer()
        dataset = load_dataset("emozilla/pg19-test", split="test")
        
        # Prepare prompts
        prompts = []
        for i in range(min(NUM_SAMPLES, len(dataset))):
            text = dataset[i]["text"]
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
        
        # Cleanup
        del llm
        
    except Exception as e:
        result = {{**exp, 'status': 'error', 'error': str(e)}}
    
    return result

def main():
    results = []
    for exp in EXPERIMENTS:
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
    cluster_name = f"roofline-{gpus_per_node}gpu-{num_nodes}node"
    local_results = Path(f"results_{cluster_name}.json")
    
    print(f"\n{'='*70}")
    print(f"Cluster: {cluster_name}")
    print(f"Config: {gpus_per_node} GPUs/node × {num_nodes} nodes")
    print(f"Experiments: {len(experiments)}")
    print("="*70)
    
    if dry_run:
        print("[DRY RUN] Would launch and run:")
        for exp in experiments[:3]:
            print(f"  - TP={exp['tp']}, PP={exp['pp']}, "
                  f"in={exp['max_input_length']}, out={exp['max_output_length']}")
        if len(experiments) > 3:
            print(f"  ... and {len(experiments)-3} more")
        return []
    
    # 1. Write YAML
    work_dir = Path.home() / "roofline_benchmarks"
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
        
        # 4. Run benchmarks
        print(f"🔥 Running benchmarks...")
        subprocess.run([
            "sky", "exec", cluster_name, 
            "--num-nodes", str(num_nodes),
            "--gpus", f"L40S:{gpus_per_node}",
            "--workdir", str(script_path.parent),
            f"source ~/sky_workdir/.venv/bin/activate && python {script_path.name}"
        ], check=True)
        
        # 5. Fetch results
        print(f"📥 Fetching results...")
        subprocess.run([
            "sky", "rsync",  'down',
            f"{cluster_name}:/tmp/benchmark_results.json",  
            str(local_results)
        ], check=True)
        
        print(f"✅ Results saved to {local_results}")
        with open(local_results) as f:
            return json.load(f)
        
    except Exception as e:
        # Mark ALL experiments in this cluster as failed
        print(f"❌ Cluster error: {e}")
        failed_results = [{**exp, 'status': 'cluster_error', 'error': str(e)} for exp in experiments]
        with open(local_results, 'w') as f:
            json.dump(failed_results, f, indent=2)
        return failed_results
        
    finally:
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
    for cluster_config, exps in sorted(groups.items()):
        results = run_cluster_benchmarks(cluster_config, exps, dry_run=dry_run)
        if results:
            all_results.extend(results)
        break
    
    # Save combined results to CSV
    if all_results:
        save_results_csv(all_results)


if __name__ == "__main__":
    main()
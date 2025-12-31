#!/usr/bin/env python3
import os
# LMCache environment variables
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# CPU Backend
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "30.0"
# SSD backend
os.environ['LMCACHE_LOCAL_DISK'] = "/tmp/lmcache_disk"
os.environ['LMCACHE_MAX_LOCAL_DISK_SIZE'] = "300"
os.environ["LMCACHE_DISK_PERSISTENCE"] = "True"
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
os.environ["LMCACHE_LOG_LEVEL"] = "WARNING"


import json
import time
import ray
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import shutil
import subprocess
import time

# Clean up existing cache directory
cache_dir = "/tmp/lmcache_disk"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

os.makedirs(cache_dir, exist_ok=True)

EXPERIMENTS = [
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 4096,
    "max_output_length": 1024,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 4096,
    "max_output_length": 4096,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 4096,
    "max_output_length": 8192,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 10000,
    "max_output_length": 1024,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 10000,
    "max_output_length": 4096,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 10000,
    "max_output_length": 8192,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 32000,
    "max_output_length": 1024,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 32000,
    "max_output_length": 4096,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 4,
    "pp": 1,
    "max_input_length": 32000,
    "max_output_length": 8192,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  }
]
RESULTS_FILE = "/tmp/benchmark_results.json"
NUM_SAMPLES = 30

def restart_ray_cluster():
    """Drop and re-create the Ray cluster before the next distributed run."""
    if os.environ.get("SKYPILOT_NODE_RANK") != "0":
        return  # only head node manages cluster lifetime
    ray.shutdown()  # detach vLLM’s Ray client first
    subprocess.run(
        "~/sky_templates/ray/stop_cluster || true",
        shell=True,
        check=False,
    )
    subprocess.run(
        "~/sky_templates/ray/start_cluster",
        shell=True,
        check=True,
    )
    time.sleep(5)  # give workers time to rejoin

def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\n{'='*70}")
    print(f"Running: TP={exp['tp']}, PP={exp['pp']}, "
          f"input={exp['max_input_length']}, output={exp['max_output_length']}")
    print("="*70)
    
    try:
        backend = "ray" if (exp['pp'] > 1) else None
        if backend == "ray":
            restart_ray_cluster()
        
        # Enable LMCache for KV cache management
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            kv_buffer_device="cpu"
        )

        llm = LLM(
            model=exp['model'],
            tensor_parallel_size=exp['tp'],
            pipeline_parallel_size=exp['pp'],
            max_model_len=min(exp['max_input_length'] + exp['max_output_length'], 32768),
            trust_remote_code=True,
            distributed_executor_backend=backend,
            gpu_memory_utilization=0.90,
            # quantization="awq",
            enforce_eager=True,
            kv_transfer_config=ktc,
            enable_prefix_caching=True
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
        
        # Cleanup BEFORE creating result
        del llm
        
        # Clean up lmcache backend
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            LMCacheEngineBuilder.destroy(ENGINE_NAME)
        except:
            if backend == "ray":
                restart_ray_cluster()
            pass
        
        # Force GPU cleanup
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        result = {
            **exp,
            'elapsed_time': elapsed,
            'total_prompt_tokens': total_prompt_tokens,
            'total_output_tokens': total_output_tokens,
            'requests_per_sec': len(outputs) / elapsed,
            'output_tokens_per_sec': total_output_tokens / elapsed,
            'total_tokens_per_sec': (total_prompt_tokens + total_output_tokens) / elapsed,
            'status': 'success'
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            error_msg = f"OOM: {error_msg[:200]}"
        if backend == "ray":
            restart_ray_cluster()
        result = {**exp, 'status': 'error', 'error': error_msg}
        
        # Force GPU cleanup on error
        try:
            import torch
            import gc
            if 'llm' in locals():
                del llm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass
    return result

def main():
    results = []
    for exp in EXPERIMENTS:
        result = run_benchmark(exp)
        results.append(result)
        print(f"Result: {result}")
        
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n✅ All done! Results in {RESULTS_FILE}")

if __name__ == "__main__":
    main()

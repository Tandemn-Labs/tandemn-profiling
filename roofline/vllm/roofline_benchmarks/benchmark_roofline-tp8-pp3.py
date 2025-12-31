#!/usr/bin/env python3
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
    "pp": 3,
    "max_input_length": 4096,
    "max_output_length": 1024,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 4096,
    "max_output_length": 4096,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 4096,
    "max_output_length": 7000,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 10000,
    "max_output_length": 1024,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 10000,
    "max_output_length": 4096,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 10000,
    "max_output_length": 7000,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 30000,
    "max_output_length": 1024,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 30000,
    "max_output_length": 4096,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  },
  {
    "tp": 8,
    "pp": 3,
    "max_input_length": 30000,
    "max_output_length": 7000,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  }
]
RESULTS_FILE = "/tmp/benchmark_results.json"
NUM_SAMPLES = 30


def run_benchmark(exp):
    """Run single benchmark experiment."""
    print(f"\n{'='*70}")
    print(f"Running: TP={exp['tp']}, PP={exp['pp']}, "
          f"input={exp['max_input_length']}, output={exp['max_output_length']}")
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
        send_discord_message(f"✅ Results saved to {exp['tp']}-{exp['pp']}")
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
        send_discord_message(f"❌ Cluster {exp['tp']}-{exp['pp']} FAILED: {error_msg}")
        # if backend == "ray":
        #     restart_ray_cluster()
        result = {**exp, 'status': 'error', 'error': error_msg}
        
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
        if i in [0,1,2,5,6,7,8,9,10]:
            continue
        result = run_benchmark(exp)
        results.append(result)
        print(f"Result: {result}")
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n✅ All done! Results in {RESULTS_FILE}")

if __name__ == "__main__":
    main()

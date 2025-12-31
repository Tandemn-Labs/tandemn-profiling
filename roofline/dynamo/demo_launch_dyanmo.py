import os
import time
import json
import subprocess
import requests
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
MODEL_PATH = "Qwen/Qwen3-0.6B"
TP_SIZE = 1
PP_SIZE = 1
INPUT_LENGTH = 512
OUTPUT_LENGTH = 512
NUM_SAMPLES = 100
DYNAMO_API_URL = "http://localhost:8000"

print("="*70)
print("🚀 DYNAMO + TensorRT-LLM BENCHMARK SETUP")
print("="*70)

# Step 1: Generate TensorRT-LLM engine config
print("\n📝 Generating TensorRT-LLM engine config...")
max_seq_len = INPUT_LENGTH + OUTPUT_LENGTH
engine_config = f"""backend: pytorch
tensor_parallel_size: {TP_SIZE}
pipeline_parallel_size: {PP_SIZE}
moe_expert_parallel_size: 1
enable_attention_dp: false

max_batch_size: 30
max_num_tokens: {max_seq_len}
max_seq_len: {max_seq_len}

kv_cache_config:
  free_gpu_memory_fraction: 0.85
  enable_block_reuse: true

cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64]

print_iter_log: true
trust_remote_code: true
"""

config_path = "/tmp/engine_config.yaml"
Path(config_path).write_text(engine_config)
print(f"   ✅ Config saved to {config_path}")

# Step 2: Start Dynamo frontend
print("\n🌐 Starting Dynamo frontend...")
frontend_proc = subprocess.Popen(
    ["python", "-m", "dynamo.frontend", "--http-port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
time.sleep(5)
print("   ✅ Frontend started on port 8000")

# Step 3: Start TensorRT-LLM worker
print(f"\n🔧 Starting TensorRT-LLM worker (TP={TP_SIZE}, PP={PP_SIZE})...")
worker_cmd = [
    "python", "-m", "dynamo.trtllm",
    "--model-path", MODEL_PATH,
    "--served-model-name", MODEL_PATH,
    "--tensor-parallel-size", str(TP_SIZE),
    "--pipeline-parallel-size", str(PP_SIZE),
    "--extra-engine-args", config_path
]

worker_proc = subprocess.Popen(
    worker_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Step 4: Wait for model to be ready
print("\n⏳ Waiting for model to be ready...")
max_wait = 300  # 5 minutes
start_wait = time.time()
model_ready = False

while time.time() - start_wait < max_wait:
    try:
        resp = requests.get(f"{DYNAMO_API_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if any(MODEL_PATH in m.get("id", "") for m in models):
                model_ready = True
                print(f"   ✅ Model ready! (took {time.time() - start_wait:.1f}s)")
                break
    except:
        pass
    time.sleep(5)
    print(f"   Still waiting... ({int(time.time() - start_wait)}s)")

if not model_ready:
    print("❌ Model failed to become ready!")
    frontend_proc.terminate()
    worker_proc.terminate()
    exit(1)

# Step 5: Load dataset and prepare prompts
print(f"\n📚 Loading pg19-test dataset...")
dataset = load_dataset("emozilla/pg19-test", split="test")
print(f"   Loaded {len(dataset)} samples from pg19-test")

print(f"\n📝 Preparing {NUM_SAMPLES} prompts (truncating to {INPUT_LENGTH} tokens each)...")
prompts = []


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

for i in range(min(NUM_SAMPLES, len(dataset))):
    text = dataset[i]["text"]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    truncated_tokens = tokens[:INPUT_LENGTH]
    truncated_text = tokenizer.decode(truncated_tokens)
    prompt = f"You are a helpful assistant. Please help me summarize the following text..\n{truncated_text}"
    prompts.append(prompt)
    
    if (i + 1) % 20 == 0:
        print(f"   Processed {i + 1}/{NUM_SAMPLES} prompts...")

print(f"✅ Prepared {len(prompts)} prompts\n")

# Step 6: Run benchmark
print("="*70)
print("🏃 STARTING BENCHMARK")
print("="*70)
print(f"Model: {MODEL_PATH}")
print(f"Tensor Parallel: {TP_SIZE}, Pipeline Parallel: {PP_SIZE}")
print(f"Number of requests: {NUM_SAMPLES}")
print(f"Input tokens per request: {INPUT_LENGTH}")
print(f"Output tokens per request: {OUTPUT_LENGTH}")
print("="*70 + "\n")

results = []
total_prompt_tokens = 0
total_output_tokens = 0

start_time = time.perf_counter()

for i, prompt in enumerate(prompts):
    try:
        resp = requests.post(
            f"{DYNAMO_API_URL}/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": OUTPUT_LENGTH,
                "min_tokens": OUTPUT_LENGTH,
                "temperature": 0.8,
                "stream": False
            },
            timeout=300
        )
        
        if resp.status_code == 200:
            result = resp.json()
            usage = result.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)
            results.append(result)
        else:
            print(f"   ⚠️  Request {i+1} failed: {resp.status_code}")
    
    except Exception as e:
        print(f"   ❌ Request {i+1} error: {str(e)[:100]}")
    
    if (i + 1) % 10 == 0:
        print(f"   Completed {i + 1}/{NUM_SAMPLES} requests...")

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Calculate metrics
total_tokens = total_prompt_tokens + total_output_tokens
requests_per_second = len(results) / elapsed_time
overall_throughput = total_tokens / elapsed_time
input_throughput = total_prompt_tokens / elapsed_time
output_throughput = total_output_tokens / elapsed_time

# Print results
print("\n" + "="*70)
print("📊 BENCHMARK RESULTS")
print("="*70)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Successful requests: {len(results)}/{NUM_SAMPLES}")
print(f"Total prompt tokens: {total_prompt_tokens:,}")
print(f"Total output tokens: {total_output_tokens:,}")
print(f"Total tokens (input + output): {total_tokens:,}")
print(f"\n🚀 THROUGHPUT: (API Level metrics, not true throughput)")
print(f"   Requests/s: {requests_per_second:.2f}")
print(f"   Overall: {overall_throughput:.2f} tokens/s")
print(f"   Input (prefill): {input_throughput:.2f} tokens/s")
print(f"   Output (decode): {output_throughput:.2f} tokens/s")

# Verify output lengths
if results:
    print(f"\n🔍 Verifying output lengths...")
    actual_lengths = []
    for result in results[:10]:
        choices = result.get("choices", [])
        if choices:
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            actual_lengths.append(completion_tokens)
    print(f"   First 10 output lengths: {actual_lengths}")
    print(f"   All should be ~{OUTPUT_LENGTH} tokens")
    
    # Sample outputs
    print(f"\n📄 Sample outputs (first 3):")
    for i in range(min(3, len(results))):
        choices = results[i].get("choices", [])
        if choices:
            output_text = choices[0].get("message", {}).get("content", "")
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt (first 100 chars): {prompts[i][:100]}...")
            print(f"Generated (first 200 chars): {output_text[:200]}...")

print("="*70)
print("\n✅ Benchmark complete!")

# Cleanup
print("\n🧹 Cleaning up...")
frontend_proc.terminate()
worker_proc.terminate()
time.sleep(2)
frontend_proc.kill()
worker_proc.kill()
print("   ✅ Processes terminated")
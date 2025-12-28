import os
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"

import ray
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams
# ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False)


llm = LLM(
    model="Qwen/Qwen3-0.6B",  # Change to your desired model
    pipeline_parallel_size=2,  # Use both GPUs
    tensor_parallel_size=1,    # Explicitly set TP=1
    dtype="auto",
    distributed_executor_backend="ray",
    max_model_len=4096,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

tokenizer = llm.get_tokenizer()
print("📚 Loading pg19-test dataset...")
dataset = load_dataset("emozilla/pg19-test", split="test")
print(f"   Loaded {len(dataset)} samples from pg19-test")
INPUT_LENGTH = 512
OUTPUT_LENGTH = 512
NUM_SAMPLES = 100

print(f"\n📝 Preparing {NUM_SAMPLES} prompts (truncating to {INPUT_LENGTH} tokens each)...")
prompts = []

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

# Sampling params: force exactly 512 output tokens, ignore EOS
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=OUTPUT_LENGTH,
    min_tokens=OUTPUT_LENGTH,  # Force minimum = maximum
    ignore_eos=True,  # Don't stop at EOS token
)

# Run benchmark
print("="*70)
print("🏃 STARTING BENCHMARK")
print("="*70)
print(f"Model: Qwen/Qwen3-0.6B")
print(f"Pipeline Parallel: 2 GPUs")
print(f"Number of requests: {NUM_SAMPLES}")
print(f"Input tokens per request: {INPUT_LENGTH}")
print(f"Output tokens per request: {OUTPUT_LENGTH}")
print("="*70 + "\n")

start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()

# Calculate metrics using vLLM's native way
elapsed_time = end_time - start_time

# Count tokens from actual outputs (vLLM native way)
total_prompt_tokens = 0
total_output_tokens = 0

for output in outputs:
    # Count actual prompt tokens from the output
    if output.prompt_token_ids:
        total_prompt_tokens += len(output.prompt_token_ids)
    
    # Count actual generated tokens from each completion output
    for completion_output in output.outputs:
        total_output_tokens += len(completion_output.token_ids)

total_tokens = total_prompt_tokens + total_output_tokens

# Calculate throughput (vLLM native approach)
requests_per_second = len(outputs) / elapsed_time
overall_throughput = total_tokens / elapsed_time
input_throughput = total_prompt_tokens / elapsed_time
output_throughput = total_output_tokens / elapsed_time

# Print results
print("\n" + "="*70)
print("📊 BENCHMARK RESULTS (Using vLLM Native Metrics)")
print("="*70)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Number of requests: {len(outputs)}")
print(f"Total prompt tokens: {total_prompt_tokens:,}")
print(f"Total output tokens: {total_output_tokens:,}")
print(f"Total tokens (input + output): {total_tokens:,}")
print(f"\n🚀 THROUGHPUT:")
print(f"   Requests/s: {requests_per_second:.2f}")
print(f"   Overall: {overall_throughput:.2f} tokens/s")
print(f"   Input (prefill): {input_throughput:.2f} tokens/s")
print(f"   Output (decode): {output_throughput:.2f} tokens/s")

# If you want detailed per-request timing metrics:
print(f"\n⏱️  DETAILED TIMING (from vLLM RequestMetrics):")
if outputs and outputs[0].metrics:
    # Example: show metrics for first request
    first_metrics = outputs[0].metrics
    print(f"   First request arrival time: {first_metrics.arrival_time:.4f}s")
    if first_metrics.first_token_time:
        ttft = first_metrics.first_token_time - first_metrics.arrival_time
        print(f"   Time to first token (TTFT): {ttft*1000:.2f} ms")
    if first_metrics.finished_time:
        total_time = first_metrics.finished_time - first_metrics.arrival_time
        print(f"   Total request time: {total_time*1000:.2f} ms")
    
    # Calculate average TTFT across all requests
    ttfts = []
    for output in outputs:
        if output.metrics and output.metrics.first_token_time:
            ttft = output.metrics.first_token_time - output.metrics.arrival_time
            ttfts.append(ttft)
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)
        print(f"   Average TTFT: {avg_ttft*1000:.2f} ms")

print("="*70)

# Verify output lengths
print(f"\n🔍 Verifying output lengths...")
actual_lengths = [len(output.outputs[0].token_ids) for output in outputs[:10]]
print(f"   First 10 output lengths: {actual_lengths}")
print(f"   All should be ~{OUTPUT_LENGTH} tokens")

# Optional: Save first few examples
print(f"\n📄 Sample outputs (first 3):")
for i in range(min(3, len(outputs))):
    output_text = outputs[i].outputs[0].text
    print(f"\n--- Sample {i+1} ---")
    print(f"Prompt (first 100 chars): {prompts[i][:100]}...")
    print(f"Generated (first 200 chars): {output_text[:200]}...")

print("\n✅ Benchmark complete!")
ray.shutdown()
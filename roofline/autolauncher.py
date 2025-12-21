#!/usr/bin/env python3

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
print(f"Total tokens: {NUM_SAMPLES * (INPUT_LENGTH + OUTPUT_LENGTH):,}")
print("="*70 + "\n")

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

# Calculate metrics
total_time = end_time - start_time
total_input_tokens = NUM_SAMPLES * INPUT_LENGTH
total_output_tokens = NUM_SAMPLES * OUTPUT_LENGTH
total_tokens = total_input_tokens + total_output_tokens

overall_throughput = total_tokens / total_time
input_throughput = total_input_tokens / total_time
output_throughput = total_output_tokens / total_time
avg_latency_per_request = total_time / NUM_SAMPLES
avg_time_per_output_token = total_time / total_output_tokens

# Print results
print("\n" + "="*70)
print("📊 BENCHMARK RESULTS")
print("="*70)
print(f"Total time: {total_time:.2f} seconds")
print(f"Number of requests: {NUM_SAMPLES}")
print(f"Total input tokens: {total_input_tokens:,}")
print(f"Total output tokens: {total_output_tokens:,}")
print(f"Total tokens (input + output): {total_tokens:,}")
print(f"\n🚀 THROUGHPUT:")
print(f"   Overall: {overall_throughput:.2f} tokens/sec")
print(f"   Input (prefill): {input_throughput:.2f} tokens/sec")
print(f"   Output (decode): {output_throughput:.2f} tokens/sec")
print(f"\n⏱️  LATENCY:")
print(f"   Avg per request: {avg_latency_per_request*1000:.2f} ms")
print(f"   Avg per output token (TPOT): {avg_time_per_output_token*1000:.2f} ms")
print("="*70)

# Verify output lengths
print(f"\n🔍 Verifying output lengths...")
actual_lengths = [len(tokenizer.encode(o.outputs[0].text)) for o in outputs[:10]]
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
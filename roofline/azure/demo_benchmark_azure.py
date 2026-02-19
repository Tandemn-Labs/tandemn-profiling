#!/usr/bin/env python3
"""
Azure Demo Benchmark — Batched vLLM inference on a single T4 GPU.
Sends a batch of summarization requests and prints throughput metrics.
"""
import json
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ── Config ──────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-0.6B"
INPUT_LENGTH = 512       # tokens per prompt
OUTPUT_LENGTH = 256      # tokens to generate
NUM_SAMPLES = 50         # number of batched requests
RESULTS_FILE = "/tmp/azure_demo_results.json"

# ── Load model ──────────────────────────────────────────────────────
print(f"Loading model: {MODEL}")
llm = LLM(
    model=MODEL,
    dtype="auto",
    max_model_len=2048,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)

# ── Prepare batched prompts ─────────────────────────────────────────
tokenizer = llm.get_tokenizer()
print(f"Loading dataset (pg19-test)...")
dataset = load_dataset("emozilla/pg19-test", split="test")

prompts = []
for i in range(min(NUM_SAMPLES, len(dataset))):
    text = dataset[i]["text"]
    tokens = tokenizer.encode(text, add_special_tokens=False)[:INPUT_LENGTH]
    truncated = tokenizer.decode(tokens)
    prompts.append(f"Please summarize the following text:\n{truncated}")

print(f"Prepared {len(prompts)} prompts ({INPUT_LENGTH} input tokens each)")

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=OUTPUT_LENGTH,
    min_tokens=OUTPUT_LENGTH,
    ignore_eos=True,
)

# ── Run batched inference ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STARTING BATCHED INFERENCE")
print("=" * 60)
print(f"  Model:            {MODEL}")
print(f"  Requests:         {NUM_SAMPLES}")
print(f"  Input tokens:     {INPUT_LENGTH}")
print(f"  Output tokens:    {OUTPUT_LENGTH}")
print("=" * 60 + "\n")

start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.perf_counter() - start

# ── Compute metrics ─────────────────────────────────────────────────
total_prompt_tokens = sum(len(o.prompt_token_ids or []) for o in outputs)
total_output_tokens = sum(
    sum(len(c.token_ids) for c in o.outputs) for o in outputs
)
total_tokens = total_prompt_tokens + total_output_tokens

requests_per_sec = len(outputs) / elapsed
output_throughput = total_output_tokens / elapsed
total_throughput = total_tokens / elapsed

# TTFT
ttfts = []
for o in outputs:
    if o.metrics and o.metrics.first_token_time:
        ttfts.append(o.metrics.first_token_time - o.metrics.arrival_time)
avg_ttft_ms = (sum(ttfts) / len(ttfts) * 1000) if ttfts else 0

# ── Print results ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"  Elapsed time:         {elapsed:.2f}s")
print(f"  Requests completed:   {len(outputs)}")
print(f"  Total prompt tokens:  {total_prompt_tokens:,}")
print(f"  Total output tokens:  {total_output_tokens:,}")
print(f"")
print(f"  Requests/s:           {requests_per_sec:.2f}")
print(f"  Output throughput:    {output_throughput:.2f} tokens/s")
print(f"  Total throughput:     {total_throughput:.2f} tokens/s")
print(f"  Avg TTFT:             {avg_ttft_ms:.2f} ms")
print("=" * 60)

# Verify output lengths
actual_lengths = [len(o.outputs[0].token_ids) for o in outputs[:5]]
print(f"\nFirst 5 output lengths: {actual_lengths} (target: {OUTPUT_LENGTH})")

# ── Save results ────────────────────────────────────────────────────
result = {
    "cloud": "azure",
    "region": "eastus",
    "gpu": "T4",
    "model": MODEL,
    "num_requests": len(outputs),
    "input_length": INPUT_LENGTH,
    "output_length": OUTPUT_LENGTH,
    "elapsed_time": elapsed,
    "total_prompt_tokens": total_prompt_tokens,
    "total_output_tokens": total_output_tokens,
    "requests_per_sec": requests_per_sec,
    "output_tokens_per_sec": output_throughput,
    "total_tokens_per_sec": total_throughput,
    "avg_ttft_ms": avg_ttft_ms,
}

with open(RESULTS_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")
print("Done!")

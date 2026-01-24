#!/usr/bin/env python3
"""
SAM2 API Benchmark - Measure latency and throughput via HTTP endpoint
"""

import time
import json
import base64
import argparse
import asyncio
import aiohttp
import numpy as np
from PIL import Image

# Config
IMAGE_PATH = "01.jpeg"
RESULTS_FILE = "sam2_api_benchmark_results.json"
DEFAULT_ENDPOINT = "http://localhost:8080"
DEFAULT_ITERATIONS = 100
DEFAULT_CONCURRENT = 1

# Prompts for testing
PROMPTS = [
    [
        {"point": (400, 350), "label": 1},
        {"point": (580, 320), "label": 1},
        {"point": (750, 330), "label": 1},
        {"point": (300, 380), "label": 1},
        {"point": (650, 290), "label": 1},
    ],
    # [
    #     {"point": (200, 100), "label": 1},
    #     {"point": (850, 150), "label": 1},
    # ],
]


def encode_image(path: str) -> str:
    """Load and encode image to base64."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def call_api(session, endpoint: str, image_b64: str, prompts: list) -> dict:
    """Call SAM2 API and return timing results."""
    payload = {"image": image_b64, "prompts": prompts, "multimask_output": False}
    start = time.perf_counter()
    
    try:
        async with session.post(
            f"{endpoint}/predict",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            elapsed = time.perf_counter() - start
            
            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "total_time": elapsed,
                    "decode_time": data.get("decode_time", 0) / 1000,
                    "encode_time": data.get("encode_time", 0) / 1000,
                    "predict_time": data.get("predict_time", 0) / 1000,
                    "gpu_time": data.get("gpu_time", 0) / 1000,
                    "rle_time": data.get("rle_time", 0) / 1000,
                    "queue_wait_ms": data.get("queue_wait_ms", 0) / 1000,
                    "num_prompts": data.get("num_prompts", 0),
                    "masks": sum(r["num_masks"] for r in data.get("results", [])),
                }
            else:
                return {"success": False, "total_time": elapsed, "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"success": False, "total_time": time.perf_counter() - start, "error": str(e)}


async def run_benchmark(endpoint: str, image_b64: str, iterations: int, concurrent: int):
    """Run benchmark with concurrent requests."""
    print(f"\nRunning {iterations} iterations, {concurrent} concurrent requests...")
    print("=" * 60)
    
    start = time.perf_counter()
    
    # Create all tasks
    tasks = [(i, p) for i in range(iterations) for p in PROMPTS]
    results = [{"iteration": i} for i in range(iterations)]
    
    async with aiohttp.ClientSession() as session:
        # Process in batches
        for batch_idx in range(0, len(tasks), concurrent):
            batch = tasks[batch_idx:batch_idx + concurrent]
            batch_results = await asyncio.gather(*[
                call_api(session, endpoint, image_b64, prompts) for _, prompts in batch
            ])
            
            # Store results
            for idx, result in enumerate(batch_results):
                task_idx = batch_idx + idx
                iter_num, prompts = tasks[task_idx]
                key = f"req{PROMPTS.index(prompts) + 1}"
                results[iter_num][key] = result
            
            if (batch_idx + len(batch)) % (concurrent * 4) == 0:
                print(f"  Progress: {batch_idx + len(batch)}/{len(tasks)} requests")
    
    return results, time.perf_counter() - start


def compute_stats(results: list, total_time: float) -> dict:
    """Compute statistics from results."""
    successful = [r for r in results if all(r.get(f"req{i+1}", {}).get("success") for i in range(len(PROMPTS)))]
    
    if not successful:
        return {"error": "No successful requests", "success_rate": 0.0}
    
    # Gather all timing data
    times = [r[f"req{i+1}"]["total_time"] for r in successful for i in range(len(PROMPTS))]
    decode_times = [r[f"req{i+1}"].get("decode_time", 0) for r in successful for i in range(len(PROMPTS))]
    encode_times = [r[f"req{i+1}"]["encode_time"] for r in successful for i in range(len(PROMPTS))]
    predict_times = [r[f"req{i+1}"]["predict_time"] for r in successful for i in range(len(PROMPTS))]
    gpu_times = [r[f"req{i+1}"].get("gpu_time", 0) for r in successful for i in range(len(PROMPTS))]
    rle_times = [r[f"req{i+1}"].get("rle_time", 0) for r in successful for i in range(len(PROMPTS))]
    queue_waits = [r[f"req{i+1}"].get("queue_wait_ms", 0) for r in successful for i in range(len(PROMPTS))]
    
    api_calls = len(successful) * len(PROMPTS)
    prompts_total = sum(r[f"req{i+1}"]["num_prompts"] for r in successful for i in range(len(PROMPTS)))
    masks_total = sum(r[f"req{i+1}"]["masks"] for r in successful for i in range(len(PROMPTS)))
    
    return {
        "iterations": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "success_rate": len(successful) / len(results) * 100,
        "total_time": total_time,
        "api_calls": api_calls,
        "prompts": prompts_total,
        "masks": masks_total,
        "calls_per_sec": api_calls / total_time,
        "prompts_per_hour": (prompts_total / total_time) * 3600,
        "avg_latency_ms": np.mean(times) * 1000,
        "p50_latency_ms": np.percentile(times, 50) * 1000,
        "p95_latency_ms": np.percentile(times, 95) * 1000,
        "p99_latency_ms": np.percentile(times, 99) * 1000,
        "avg_decode_ms": np.mean(decode_times) * 1000,
        "avg_encode_ms": np.mean(encode_times) * 1000,
        "avg_predict_ms": np.mean(predict_times) * 1000,
        "avg_gpu_ms": np.mean(gpu_times) * 1000,
        "avg_rle_ms": np.mean(rle_times) * 1000,
        "avg_queue_wait_ms": np.mean(queue_waits) * 1000,
    }


def print_results(stats: dict):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Time: {stats['total_time']:.1f}s")
    print(f"Success: {stats['successful']}/{stats['iterations']} ({stats['success_rate']:.1f}%)")
    
    if stats.get('error'):
        return
    
    print(f"\nTHROUGHPUT:")
    print(f"  {stats['calls_per_sec']:.1f} calls/sec")
    print(f"  {stats['prompts_per_hour']:.0f} prompts/hour")
    
    print(f"\nLATENCY:")
    print(f"  Avg: {stats['avg_latency_ms']:.1f}ms")
    print(f"  P50: {stats['p50_latency_ms']:.1f}ms")
    print(f"  P95: {stats['p95_latency_ms']:.1f}ms")
    print(f"  P99: {stats['p99_latency_ms']:.1f}ms")
    
    print(f"\nBREAKDOWN:")
    print(f"  Queue Wait: {stats['avg_queue_wait_ms']:.1f}ms")
    print(f"  Decode (CPU): {stats['avg_decode_ms']:.1f}ms")
    print(f"  Encode (GPU): {stats['avg_encode_ms']:.1f}ms")
    print(f"  Predict (GPU): {stats['avg_predict_ms']:.1f}ms")
    print(f"  GPU Total: {stats['avg_gpu_ms']:.1f}ms")
    print(f"  RLE (CPU): {stats['avg_rle_ms']:.1f}ms")


async def main():
    parser = argparse.ArgumentParser(description='SAM2 API Benchmark')
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINT, help='API endpoint URL')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, help='Number of iterations')
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT, help='Concurrent requests')
    parser.add_argument('--image', default=IMAGE_PATH, help='Path to test image')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM2 API Benchmark")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print(f"Iterations: {args.iterations}")
    print(f"Concurrent: {args.concurrent}")
    
    # Load image
    try:
        img = Image.open(args.image)
        print(f"Image: {args.image} ({img.size[0]}x{img.size[1]})")
        image_b64 = encode_image(args.image)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return 1
    
    # Run benchmark
    results, total_time = await run_benchmark(args.endpoint, image_b64, args.iterations, args.concurrent)
    stats = compute_stats(results, total_time)
    print_results(stats)
    
    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump({"config": vars(args), "stats": stats, "results": results}, f, indent=2)
    
    print(f"\n✅ Saved to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

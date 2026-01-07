#!/usr/bin/env python3
"""
SAM2 Roofline Benchmark
-----------------------
Simple, hackable benchmark to measure SAM2 throughput on T4.
Measures: images/hour, prompts/hour, masks/second
"""

import time
import json
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =============================================================================
# CONFIG - Easy to hack!
# =============================================================================

IMAGE_PATH = "01.jpg"  # In workdir root
CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"  # Created in setup
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # From sam2 package
RESULTS_FILE = "/tmp/sam2_benchmark_results.json"  # Or just "results.json"

# Number of benchmark iterations
NUM_ITERATIONS = 100  # How many times to repeat the full test

# Prompts: (x, y) point coordinates for the highway image
# These are cars in the image - feel free to add more!
PROMPTS_IMAGE_1 = [
    {"point": (400, 350), "label": 1},   # white car left lane
    {"point": (580, 320), "label": 1},   # dark car center
    {"point": (750, 330), "label": 1},   # car right side
    {"point": (300, 380), "label": 1},   # car in merge lane
    {"point": (650, 290), "label": 1},   # car far ahead
]

# Second "image" - same image, different prompts (simulating multiple images)
PROMPTS_IMAGE_2 = [
    {"point": (200, 100), "label": 1},   # trees top left
    {"point": (850, 150), "label": 1},   # building right
    {"point": (500, 450), "label": 1},   # road surface
    {"point": (100, 300), "label": 1},   # guardrail/barrier
    {"point": (450, 50), "label": 1},    # sky/bridge
]

# =============================================================================
# BENCHMARK CODE
# =============================================================================

def load_model():
    """Load SAM2 model."""
    print("Loading SAM2 model...")
    start = time.perf_counter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2)
    
    load_time = time.perf_counter() - start
    print(f"Model loaded in {load_time:.2f}s")
    
    return predictor


def run_prompts(predictor, image, prompts):
    """Run a batch of point prompts on an image. Returns masks and timing."""
    
    # Set the image (this does the image encoding)
    encode_start = time.perf_counter()
    predictor.set_image(image)
    encode_time = time.perf_counter() - encode_start
    
    masks_generated = 0
    predict_times = []
    
    for prompt in prompts:
        point = np.array([[prompt["point"][0], prompt["point"][1]]])
        label = np.array([prompt["label"]])
        
        pred_start = time.perf_counter()
        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True  # Returns 3 masks per prompt
        )
        pred_time = time.perf_counter() - pred_start
        
        predict_times.append(pred_time)
        masks_generated += len(masks)
    
    return {
        "encode_time": encode_time,
        "predict_times": predict_times,
        "total_predict_time": sum(predict_times),
        "masks_generated": masks_generated,
        "num_prompts": len(prompts),
    }


def run_benchmark(predictor, image):
    """Run full benchmark: 2 images × 5 prompts each × N iterations."""
    
    all_results = []
    
    print(f"\nRunning {NUM_ITERATIONS} iterations...")
    print("=" * 60)
    
    total_start = time.perf_counter()
    
    for i in range(NUM_ITERATIONS):
        iter_start = time.perf_counter()
        
        # Image 1 with 5 prompts
        result1 = run_prompts(predictor, image, PROMPTS_IMAGE_1)
        
        # Image 2 with 5 different prompts (same underlying image, simulates new image)
        result2 = run_prompts(predictor, image, PROMPTS_IMAGE_2)
        
        iter_time = time.perf_counter() - iter_start
        
        all_results.append({
            "iteration": i,
            "image1": result1,
            "image2": result2,
            "iteration_time": iter_time,
        })
        
        # Progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{NUM_ITERATIONS} - {iter_time:.3f}s per iteration")
    
    total_time = time.perf_counter() - total_start
    
    return all_results, total_time


def compute_stats(results, total_time):
    """Compute throughput statistics."""
    
    num_iterations = len(results)
    images_processed = num_iterations * 2  # 2 images per iteration
    prompts_processed = num_iterations * 10  # 5+5 prompts per iteration
    masks_generated = sum(
        r["image1"]["masks_generated"] + r["image2"]["masks_generated"]
        for r in results
    )
    
    # Average times
    avg_encode = np.mean([r["image1"]["encode_time"] for r in results])
    avg_predict_per_prompt = np.mean([
        t for r in results
        for t in r["image1"]["predict_times"] + r["image2"]["predict_times"]
    ])
    
    stats = {
        # Counts
        "total_iterations": num_iterations,
        "total_images": images_processed,
        "total_prompts": prompts_processed,
        "total_masks": masks_generated,
        "total_time_seconds": total_time,
        
        # Throughput
        "images_per_second": images_processed / total_time,
        "images_per_hour": (images_processed / total_time) * 3600,
        "prompts_per_second": prompts_processed / total_time,
        "prompts_per_hour": (prompts_processed / total_time) * 3600,
        "masks_per_second": masks_generated / total_time,
        
        # Latency
        "avg_image_encode_ms": avg_encode * 1000,
        "avg_predict_per_prompt_ms": avg_predict_per_prompt * 1000,
        "avg_iteration_ms": (total_time / num_iterations) * 1000,
    }
    
    return stats


def main():
    print("=" * 60)
    print("SAM2 Roofline Benchmark")
    print("=" * 60)
    
    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load image
    print(f"\nLoading image: {IMAGE_PATH}")
    image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    print(f"Image shape: {image.shape}")
    
    # Load model
    predictor = load_model()
    
    # Warmup (important for accurate timing!)
    print("\nWarmup run...")
    _ = run_prompts(predictor, image, PROMPTS_IMAGE_1[:2])
    torch.cuda.synchronize()
    
    # Run benchmark
    results, total_time = run_benchmark(predictor, image)
    
    # Compute stats
    stats = compute_stats(results, total_time)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"")
    print(f"📊 THROUGHPUT:")
    print(f"   Images/hour:  {stats['images_per_hour']:.0f}")
    print(f"   Prompts/hour: {stats['prompts_per_hour']:.0f}")
    print(f"   Masks/second: {stats['masks_per_second']:.1f}")
    print(f"")
    print(f"⏱️  LATENCY:")
    print(f"   Image encode:     {stats['avg_image_encode_ms']:.1f} ms")
    print(f"   Per-prompt mask:  {stats['avg_predict_per_prompt_ms']:.1f} ms")
    print(f"")
    print(f"📈 RAW COUNTS:")
    print(f"   Images processed: {stats['total_images']}")
    print(f"   Prompts run:      {stats['total_prompts']}")
    print(f"   Masks generated:  {stats['total_masks']}")
    
    # Save results
    output = {
        "config": {
            "model": MODEL_CFG,
            "checkpoint": CHECKPOINT,
            "image_path": IMAGE_PATH,
            "num_iterations": NUM_ITERATIONS,
            "prompts_per_image": 5,
        },
        "stats": stats,
        "raw_results": results,
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
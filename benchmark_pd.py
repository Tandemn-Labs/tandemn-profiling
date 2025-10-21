# benchmark_prefill_decode_ssd_storage.py
"""
Benchmark prefill-only and decode-only speeds on single GPU with SSD storage.
Uses SharedStorageConnector to save KV cache to SSD between prefill and decode.
"""

import os
import time
import json
import gc
import torch
from datetime import datetime
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from datasets import load_dataset
import argparse

billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

# Setup for vLLM v1 (required for SharedStorageConnector)
os.environ["VLLM_USE_V1"] = "1"

def read_sonnet_prompts(file_path="./benchmarks/sonnet.txt", num_prompts=10, input_length=1000):
    """Read sonnet text and create prompts"""
    # Create prompts by repeating sonnet text
    prompts = []
    for i in range(num_prompts):
        prompts.append("Summarize This: " + billsum["train"][i]["text"][:input_length])  # ~1000 chars input

    return prompts

def apply_chat_template(llm, prompts):
    """Apply chat template to prompts"""
    tokenizer = llm.get_tokenizer()

    formatted_prompts = []
    for prompt in prompts:
        # Format as chat message
        messages = [
            {"role": "user", "content": prompt}
        ]
        # Apply template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    return formatted_prompts

def benchmark_prefill_only(llm, prompts, output_file="prefill_prompts.json"):
    """Measure prefill-only speed and save prompts for decode phase"""
    print("\n" + "="*80)
    print("BENCHMARKING PREFILL-ONLY (max_tokens=1)")
    print("="*80)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1,  # Generate 1 token, saves KV cache to SSD
        ignore_eos=True
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_time = end_time - start_time
    total_input_tokens = sum(len(llm.get_tokenizer().encode(p)) for p in prompts)

    # Save prompts + first generated token for decode phase
    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
    
    with open(output_file, 'w') as f:
        json.dump({"prompts": prompts}, f, indent=2)
    
    print(f"\n📊 PREFILL-ONLY RESULTS:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Number of prompts: {len(prompts)}")
    print(f"  Total input tokens: {total_input_tokens}")
    print(f"  Prefill throughput: {total_input_tokens / total_time:.2f} tokens/sec")
    print(f"  Avg prefill latency: {(total_time / len(prompts)) * 1000:.2f} ms per prompt")
    print(f"  ✅ KV caches are now stored on SSD!")
    print(f"  💾 Prompts saved to: {output_file}")

    return {
        'total_time': total_time,
        'num_prompts': len(prompts),
        'total_tokens': total_input_tokens,
        'throughput': total_input_tokens / total_time,
        'avg_latency_ms': (total_time / len(prompts)) * 1000
    }, new_prompts

def benchmark_decode_only(llm, prompts, max_tokens=100, input_file="prefill_prompts.json"):
    """Measure decode-only speed by reusing prefill KV cache from SSD"""
    print("\n" + "="*80)
    print(f"BENCHMARKING DECODE-ONLY (max_tokens={max_tokens})")
    print("="*80)
    print("⚡ KV cache will be loaded from SSD storage, skipping prefill!")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        ignore_eos=True
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_time = end_time - start_time
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    decoded_texts = [o.outputs[0].text for o in outputs]
    # Calculate decode-only metrics (excluding prefill)
    avg_decode_time = total_time / len(prompts)
    avg_tokens_per_req = total_output_tokens / len(prompts)
    avg_tpot = (avg_decode_time / avg_tokens_per_req) * 1000 if avg_tokens_per_req > 0 else 0  # ms per token

    print(f"\n📊 DECODE-ONLY RESULTS:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Number of prompts: {len(prompts)}")
    print(f"  Total output tokens: {total_output_tokens}")
    print(f"  Decode throughput: {total_output_tokens / total_time:.2f} tokens/sec")
    print(f"  Avg TPOT (per decode token): {avg_tpot:.2f} ms")
    print(f"  Avg decode latency: {avg_decode_time * 1000:.2f} ms per prompt")

    return {
        'total_time': total_time,
        'num_prompts': len(prompts),
        'total_tokens': total_output_tokens,
        'throughput': total_output_tokens / total_time,
        'avg_tpot_ms': avg_tpot,
        'avg_latency_ms': avg_decode_time * 1000,
        'decoded_outputs': decoded_texts
    }

def save_results(config, prefill_results=None, decode_results=None, output_dir="./benchmark_results"):
    """Save benchmark results to JSON and CSV files"""

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare comprehensive results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "prefill": prefill_results,
        "decode": decode_results,
    }
    
    if prefill_results and decode_results:
        results["comparison"] = {
            "prefill_throughput_tokens_per_sec": prefill_results['throughput'],
            "decode_throughput_tokens_per_sec": decode_results['throughput'],
            "decode_to_prefill_ratio": decode_results['throughput'] / prefill_results['throughput'],
            "prefill_latency_ms": prefill_results['avg_latency_ms'],
            "decode_tpot_ms": decode_results['avg_tpot_ms']
        }

    # Save as JSON
    json_filename = f"{output_dir}/benchmark_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_filename}")

    # Save decoded outputs separately
    if decode_results and decode_results.get('decoded_outputs'):
        outputs_filename = f"{output_dir}/decoded_outputs_{timestamp}.json"
        with open(outputs_filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": config['model'],
                "num_prompts": config['num_prompts'],
                "outputs": decode_results['decoded_outputs']
            }, f, indent=2)
        print(f"📄 Decoded outputs saved to: {outputs_filename}")

    # Save as CSV (append mode for easy comparison across runs)
    csv_filename = f"{output_dir}/benchmark_summary.csv"
    file_exists = Path(csv_filename).exists()

    with open(csv_filename, 'a') as f:
        if not file_exists:
            # Write header
            f.write("timestamp,model,num_prompts,max_model_len,")
            f.write("prefill_max_batched_tokens,prefill_max_seqs,decode_max_batched_tokens,decode_max_seqs,")
            f.write("prefill_throughput,prefill_latency_ms,decode_throughput,decode_tpot_ms,decode_latency_ms\n")

        # Write data
        f.write(f"{timestamp},")
        f.write(f"{config['model']},")
        f.write(f"{config['num_prompts']},")
        f.write(f"{config['max_model_len']},")
        f.write(f"{config.get('prefill_max_num_batched_tokens', 'N/A')},")
        f.write(f"{config.get('prefill_max_num_seqs', 'N/A')},")
        f.write(f"{config.get('decode_max_num_batched_tokens', 'N/A')},")
        f.write(f"{config.get('decode_max_num_seqs', 'N/A')},")
        
        if prefill_results:
            f.write(f"{prefill_results['throughput']:.2f},")
            f.write(f"{prefill_results['avg_latency_ms']:.2f},")
        else:
            f.write("N/A,N/A,")
        
        if decode_results:
            f.write(f"{decode_results['throughput']:.2f},")
            f.write(f"{decode_results['avg_tpot_ms']:.2f},")
            f.write(f"{decode_results['avg_latency_ms']:.2f}\n")
        else:
            f.write("N/A,N/A,N/A\n")

    print(f"📊 Summary appended to: {csv_filename}")

    return json_filename, csv_filename

def load_cfg():
    """
    Load a JSON config file if provided. Priority:
    1) --cfg <path>
    2) BENCH_CFG env var
    3) ./benchmark_pd.json if it exists
    Otherwise returns {} and uses in-script defaults.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg", type=str, default=None)
    args, _ = parser.parse_known_args()

    candidate_paths = []
    if args.cfg:
        candidate_paths.append(args.cfg)
    if os.environ.get("BENCH_CFG"):
        candidate_paths.append(os.environ["BENCH_CFG"])
    default_path = "benchmark_pd.json"
    if Path(default_path).exists():
        candidate_paths.append(default_path)

    for p in candidate_paths:
        p = os.path.expanduser(p)
        if p and Path(p).exists():
            with open(p, "r") as f:
                cfg = json.load(f)
            print(f"[CFG] Loaded config from: {p}")
            return cfg

    print("[CFG] No config file found (using in-script defaults).")
    return {}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark prefill and decode with SSD KV cache storage")
    parser.add_argument("--mode", type=str, choices=["prefill-only", "decode-only", "both"], default="both",
                       help="Run mode: prefill-only, decode-only, or both")
    parser.add_argument("--cfg", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--prefill-prompts-file", type=str, default="prefill_prompts.json",
                       help="File to save/load prefill prompts")
    args = parser.parse_args()

    # Load config
    CFG = load_cfg()

    # Load configuration
    MODEL = CFG.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    NUM_PROMPTS = int(CFG.get("NUM_PROMPTS", 2000))
    MAX_MODEL_LEN = int(CFG.get("MAX_MODEL_LEN", 4096))
    PREFILL_MAX_NUM_BATCHED_TOKENS = int(CFG.get("PREFILL_MAX_NUM_BATCHED_TOKENS", 32768))
    PREFILL_MAX_NUM_SEQS = int(CFG.get("PREFILL_MAX_NUM_SEQS", 128))
    DECODE_MAX_NUM_BATCHED_TOKENS = int(CFG.get("DECODE_MAX_NUM_BATCHED_TOKENS", 65536))
    DECODE_MAX_NUM_SEQS = int(CFG.get("DECODE_MAX_NUM_SEQS", 256))
    GPU_MEMORY_UTILIZATION = float(CFG.get("GPU_MEMORY_UTILIZATION", 0.95))
    INPUT_LENGTH = int(CFG.get("INPUT_LENGTH", 1000))
    SHARED_STORAGE_PATH = CFG.get("SHARED_STORAGE_PATH", "./kv_cache_storage")

    config = {
        "model": MODEL,
        "num_prompts": NUM_PROMPTS,
        "max_model_len": MAX_MODEL_LEN,
        "prefill_max_num_batched_tokens": PREFILL_MAX_NUM_BATCHED_TOKENS,
        "prefill_max_num_seqs": PREFILL_MAX_NUM_SEQS,
        "decode_max_num_batched_tokens": DECODE_MAX_NUM_BATCHED_TOKENS,
        "decode_max_num_seqs": DECODE_MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "input_length": INPUT_LENGTH,
        "shared_storage_path": SHARED_STORAGE_PATH
    }

    print("="*80)
    print("SSD STORAGE BENCHMARK - PREFILL/DECODE SEPARATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Configure KV cache transfer to SSD
    kv_transfer_config = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": SHARED_STORAGE_PATH
        }
    )

    prefill_results = None
    decode_results = None
    new_prompts = None

    # ===== PREFILL PHASE =====
    if args.mode in ["prefill-only", "both"]:
        print("\n" + "="*80)
        print("🚀 PREFILL PHASE")
        print("="*80)
        
        # Initialize LLM for prefill
        print(f"Initializing vLLM with prefill parameters...")
        print(f"  - max_num_batched_tokens: {PREFILL_MAX_NUM_BATCHED_TOKENS}")
        print(f"  - max_num_seqs: {PREFILL_MAX_NUM_SEQS}")
        
        llm_prefill = LLM(
            model=MODEL,
            kv_transfer_config=kv_transfer_config,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_num_batched_tokens=PREFILL_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=PREFILL_MAX_NUM_SEQS,
            enable_prefix_caching=False,
            enforce_eager=True
        )

        # Load and prepare prompts
        print(f"\n📝 Loading {NUM_PROMPTS} prompts from dataset...")
        prompts = read_sonnet_prompts(num_prompts=NUM_PROMPTS, input_length=INPUT_LENGTH)
        prompts = apply_chat_template(llm_prefill, prompts)
        print(f"✅ Prepared {len(prompts)} prompts with chat template")

        # Run prefill benchmark
        prefill_results, new_prompts = benchmark_prefill_only(llm_prefill, prompts, args.prefill_prompts_file)

        # Cleanup prefill LLM with proper memory release
        print("\n🧹 Cleaning up prefill LLM...")
        
        # For non-multiprocessing mode, need manual cleanup
        if not os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "1":
            try:
                # Access internal components for manual cleanup
                llm_engine = llm_prefill.llm_engine.engine_core.engine_core
                model_runner = llm_engine.model_executor.driver_worker.worker.model_runner
                del model_runner.model
                del model_runner.kv_caches
                print("  ✓ Manually cleaned up model and KV caches")
            except Exception as e:
                print(f"  ⚠️ Manual cleanup failed: {e}")
        
        # Standard cleanup
        del llm_prefill
        gc.collect()
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  ✓ Cleared CUDA cache")
        
        # Additional cleanup for distributed environment
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory()
            print("  ✓ Cleaned up distributed environment")
        except:
            pass
        
        time.sleep(3)  # Give time for memory to be released

    # ===== DECODE PHASE =====
    if args.mode in ["decode-only", "both"]:
        print("\n" + "="*80)
        print("🚀 DECODE PHASE")
        print("="*80)

        # Check if prefill was done
        if args.mode == "decode-only":
            if not Path(args.prefill_prompts_file).exists():
                print(f"❌ ERROR: Prefill prompts file '{args.prefill_prompts_file}' not found!")
                print("   Please run with --mode prefill-only first, or use --mode both")
                return
            
            # Load prompts from file
            print(f"📂 Loading prompts from {args.prefill_prompts_file}...")
            with open(args.prefill_prompts_file, 'r') as f:
                data = json.load(f)
                new_prompts = data['prompts']
            print(f"✅ Loaded {len(new_prompts)} prompts")

        # Initialize LLM for decode with different parameters
        print(f"Initializing vLLM with decode parameters...")
        print(f"  - max_num_batched_tokens: {DECODE_MAX_NUM_BATCHED_TOKENS}")
        print(f"  - max_num_seqs: {DECODE_MAX_NUM_SEQS}")
        
        llm_decode = LLM(
            model=MODEL,
            kv_transfer_config=kv_transfer_config,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_num_batched_tokens=DECODE_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=DECODE_MAX_NUM_SEQS,
            enable_prefix_caching=False,
            enforce_eager=True
        )

        # Run decode benchmark
        decode_results = benchmark_decode_only(llm_decode, new_prompts, max_tokens=100, input_file=args.prefill_prompts_file)

        # Cleanup decode LLM with proper memory release
        print("\n🧹 Cleaning up decode LLM...")
        
        # For non-multiprocessing mode, need manual cleanup
        if not os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "1":
            try:
                # Access internal components for manual cleanup
                llm_engine = llm_decode.llm_engine.engine_core.engine_core
                model_runner = llm_engine.model_executor.driver_worker.worker.model_runner
                del model_runner.model
                del model_runner.kv_caches
                print("  ✓ Manually cleaned up model and KV caches")
            except Exception as e:
                print(f"  ⚠️ Manual cleanup failed: {e}")
        
        # Standard cleanup
        del llm_decode
        gc.collect()
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  ✓ Cleared CUDA cache")
        
        # Additional cleanup for distributed environment  
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory()
            print("  ✓ Cleaned up distributed environment")
        except:
            pass

    # ===== FINAL SUMMARY =====
    print("\n" + "="*80)
    print("📈 FINAL SUMMARY")
    print("="*80)
    
    if prefill_results:
        print(f"\n🔸 PREFILL-ONLY:")
        print(f"    Throughput: {prefill_results['throughput']:.2f} tokens/sec")
        print(f"    Avg Latency: {prefill_results['avg_latency_ms']:.2f} ms")

    if decode_results:
        print(f"\n🔸 DECODE-ONLY (with SSD KV cache):")
        print(f"    Throughput: {decode_results['throughput']:.2f} tokens/sec")
        print(f"    Avg TPOT: {decode_results['avg_tpot_ms']:.2f} ms")
        print(f"    Avg Latency: {decode_results['avg_latency_ms']:.2f} ms")

    if prefill_results and decode_results:
        print(f"\n🎯 Prefill vs Decode:")
        print(f"    Prefill speed: {prefill_results['throughput']:.2f} tokens/sec")
        print(f"    Decode speed: {decode_results['throughput']:.2f} tokens/sec")
        ratio = decode_results['throughput'] / prefill_results['throughput']
        print(f"    Ratio (decode/prefill): {ratio:.2f}x")

    # Save results to files
    json_file, csv_file = save_results(config, prefill_results, decode_results)

    print("\n✅ Benchmark complete!")
    print("="*80)

if __name__ == "__main__":
    main()

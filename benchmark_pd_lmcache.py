# benchmark_prefill_decode_lmcache_ssd.py
"""
Benchmark prefill-only and decode-only speeds on single GPU with LMCache SSD storage.
Uses LMCacheConnectorV1 to save KV cache to SSD between prefill and decode.
"""

import os
import time
import json
import gc
import torch
import signal
import sys
import atexit
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from datasets import load_dataset
import argparse
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
import tempfile
import shutil

billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

# Setup for vLLM v1 (required for LMCacheConnectorV1)
os.environ["VLLM_USE_V1"] = "1"

def setup_lmcache_environment(ssd_path: str, cache_size_gb: float = 10.0):
    """Setup LMCache environment for SSD storage"""
    # Enable experimental features
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # Set chunk size (256 tokens per chunk is recommended)
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Enable local disk storage (SSD)
    os.environ["LMCACHE_LOCAL_DISK"] = ssd_path
    # Set disk storage size limit (in GB)
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(cache_size_gb)
    # IMPORTANT: Set CPU size to small value instead of 0 to avoid buffer allocation error
    # LMCache needs a minimal CPU buffer even when using disk-only mode
    os.environ["LMCACHE_LOCAL_CPU"] = "True"  # Must be True to avoid initialization issues
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "0.1"  # Small buffer (100MB) to avoid the zero-size error
    # Enable disk persistence
    os.environ["LMCACHE_DISK_PERSISTENCE"] = "True"
    # Disable remote storage
    os.environ["LMCACHE_REMOTE_URL"] = ""
    print(f"✅ LMCache configured for SSD storage:")
    print(f"   - Storage path: {ssd_path}")
    print(f"   - Max size: {cache_size_gb} GB")
    print(f"   - Chunk size: 256 tokens")
    print(f"   - CPU buffer: 0.1 GB (minimal, for compatibility)")

def read_sonnet_prompts(file_path="./benchmarks/sonnet.txt", num_prompts=10, input_length=1000):
    """Read sonnet text and create prompts"""
    prompts = []
    for i in range(num_prompts):
        prompts.append("Summarize This: " + billsum["train"][i]["text"][:input_length])
    return prompts

def apply_chat_template(llm, prompts):
    """Apply chat template to prompts"""
    tokenizer = llm.get_tokenizer()
    
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]
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
        max_tokens=1,  # Generate 1 token, saves KV cache to SSD via LMCache
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
    print(f"  ✅ KV caches are now stored on SSD via LMCache!")
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
    print("⚡ KV cache will be loaded from SSD storage via LMCache, skipping prefill!")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True
    )
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    total_time = end_time - start_time
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    decoded_texts = [o.outputs[0].text for o in outputs]
    
    avg_decode_time = total_time / len(prompts)
    avg_tokens_per_req = total_output_tokens / len(prompts)
    avg_tpot = (avg_decode_time / avg_tokens_per_req) * 1000 if avg_tokens_per_req > 0 else 0
    
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    json_filename = f"{output_dir}/benchmark_lmcache_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_filename}")
    
    if decode_results and decode_results.get('decoded_outputs'):
        outputs_filename = f"{output_dir}/decoded_outputs_lmcache_{timestamp}.json"
        with open(outputs_filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": config['model'],
                "num_prompts": config['num_prompts'],
                "outputs": decode_results['decoded_outputs']
            }, f, indent=2)
        print(f"📄 Decoded outputs saved to: {outputs_filename}")
    
    return json_filename

def load_cfg():
    """Load configuration from JSON file"""
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

def cleanup_llm(llm):
    """Properly cleanup LLM and LMCache resources"""
    print("\n🧹 Cleaning up LLM and LMCache...")
    
    try:
        # Clean up LMCache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        print("  ✓ LMCache engine destroyed")
    except Exception as e:
        print(f"  ⚠️ LMCache cleanup warning: {e}")
    
    # Force kill any engine processes
    try:
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'engine_core'):
                # Try to terminate the engine process
                if hasattr(engine.engine_core, 'process'):
                    process = engine.engine_core.process
                    if process and process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=2)
                        print("  ✓ Terminated engine process")
    except Exception as e:
        print(f"  ⚠️ Process cleanup warning: {e}")
    
    # Standard cleanup
    del llm
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
    
    # Kill any remaining EngineCore processes
    try:
        import subprocess
        subprocess.run(['pkill', '-f', 'EngineCore'], capture_output=True, timeout=2)
    except:
        pass
    
    time.sleep(2)  # Give time for memory to be released

# Global variable to track active LLM instances for cleanup
_active_llm_instances = []

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n⚠️  Interrupt received, cleaning up...")
    # Clean up any active LLM instances
    for llm in _active_llm_instances:
        try:
            cleanup_llm(llm)
        except:
            pass
    _active_llm_instances.clear()
    sys.exit(0)

@contextmanager
def create_llm(*args, **kwargs):
    """Context manager for LLM with guaranteed cleanup"""
    global _active_llm_instances
    llm = None
    try:
        llm = LLM(*args, **kwargs)
        _active_llm_instances.append(llm)
        yield llm
    finally:
        if llm is not None:
            try:
                _active_llm_instances.remove(llm)
            except ValueError:
                pass
            cleanup_llm(llm)

def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup on exit
    atexit.register(lambda: print("✓ Cleanup complete"))
    
    parser = argparse.ArgumentParser(description="Benchmark prefill and decode with LMCache SSD storage")
    parser.add_argument("--mode", type=str, choices=["prefill-only", "decode-only", "both"], default="both",
                       help="Run mode: prefill-only, decode-only, or both")
    parser.add_argument("--cfg", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--prefill-prompts-file", type=str, default="prefill_prompts.json",
                       help="File to save/load prefill prompts")
    parser.add_argument("--ssd-path", type=str, default="./lmcache_ssd_storage",
                       help="Path to SSD storage for LMCache")
    parser.add_argument("--cache-size-gb", type=float, default=50.0,
                       help="Maximum cache size in GB on SSD")
    args = parser.parse_args()
    
    # Load config
    CFG = load_cfg()
    
    # Configuration
    MODEL = CFG.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    NUM_PROMPTS = int(CFG.get("NUM_PROMPTS", 2000))
    MAX_MODEL_LEN = int(CFG.get("MAX_MODEL_LEN", 4096))
    PREFILL_MAX_NUM_BATCHED_TOKENS = int(CFG.get("PREFILL_MAX_NUM_BATCHED_TOKENS", 32768))
    PREFILL_MAX_NUM_SEQS = int(CFG.get("PREFILL_MAX_NUM_SEQS", 128))
    DECODE_MAX_NUM_BATCHED_TOKENS = int(CFG.get("DECODE_MAX_NUM_BATCHED_TOKENS", 65536))
    DECODE_MAX_NUM_SEQS = int(CFG.get("DECODE_MAX_NUM_SEQS", 256))
    GPU_MEMORY_UTILIZATION = float(CFG.get("GPU_MEMORY_UTILIZATION", 0.95))
    INPUT_LENGTH = int(CFG.get("INPUT_LENGTH", 1000))
    
    # Ensure SSD path exists
    Path(args.ssd_path).mkdir(parents=True, exist_ok=True)
    
    # Setup LMCache environment for SSD storage
    setup_lmcache_environment(args.ssd_path, args.cache_size_gb)
    
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
        "lmcache_ssd_path": args.ssd_path,
        "lmcache_cache_size_gb": args.cache_size_gb
    }
    
    print("="*80)
    print("LMCACHE SSD STORAGE BENCHMARK - PREFILL/DECODE SEPARATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Configure KV cache transfer with LMCache
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    
    prefill_results = None
    decode_results = None
    new_prompts = None
    
    # ===== PREFILL PHASE =====
    if args.mode in ["prefill-only", "both"]:
        print("\n" + "="*80)
        print("🚀 PREFILL PHASE (with LMCache SSD storage)")
        print("="*80)
        
        print(f"Initializing vLLM with LMCache prefill parameters...")
        print(f"  - max_num_batched_tokens: {PREFILL_MAX_NUM_BATCHED_TOKENS}")
        print(f"  - max_num_seqs: {PREFILL_MAX_NUM_SEQS}")
        print(f"  - SSD storage: {args.ssd_path}")
        
        with create_llm(
            model=MODEL,
            kv_transfer_config=kv_transfer_config,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_num_batched_tokens=PREFILL_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=PREFILL_MAX_NUM_SEQS,
            enable_prefix_caching=False,
            enforce_eager=True
        ) as llm_prefill:
            
            print(f"\n📝 Loading {NUM_PROMPTS} prompts from dataset...")
            prompts = read_sonnet_prompts(num_prompts=NUM_PROMPTS, input_length=INPUT_LENGTH)
            prompts = apply_chat_template(llm_prefill, prompts)
            print(f"✅ Prepared {len(prompts)} prompts with chat template")
            
            prefill_results, new_prompts = benchmark_prefill_only(llm_prefill, prompts, args.prefill_prompts_file)
    
    # ===== DECODE PHASE =====
    if args.mode in ["decode-only", "both"]:
        print("\n" + "="*80)
        print("🚀 DECODE PHASE (with LMCache SSD storage)")
        print("="*80)
        
        if args.mode == "decode-only":
            if not Path(args.prefill_prompts_file).exists():
                print(f"❌ ERROR: Prefill prompts file '{args.prefill_prompts_file}' not found!")
                print("   Please run with --mode prefill-only first, or use --mode both")
                return
            
            print(f"📂 Loading prompts from {args.prefill_prompts_file}...")
            with open(args.prefill_prompts_file, 'r') as f:
                data = json.load(f)
                new_prompts = data['prompts']
            print(f"✅ Loaded {len(new_prompts)} prompts")
        
        print(f"Initializing vLLM with LMCache decode parameters...")
        print(f"  - max_num_batched_tokens: {DECODE_MAX_NUM_BATCHED_TOKENS}")
        print(f"  - max_num_seqs: {DECODE_MAX_NUM_SEQS}")
        print(f"  - SSD storage: {args.ssd_path}")
        
        with create_llm(
            model=MODEL,
            kv_transfer_config=kv_transfer_config,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_num_batched_tokens=DECODE_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=DECODE_MAX_NUM_SEQS,
            enable_prefix_caching=False,
            enforce_eager=True
        ) as llm_decode:
            
            decode_results = benchmark_decode_only(llm_decode, new_prompts, max_tokens=3500, input_file=args.prefill_prompts_file)
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*80)
    print("📈 FINAL SUMMARY (LMCache with SSD)")
    print("="*80)
    
    if prefill_results:
        print(f"\n🔸 PREFILL-ONLY:")
        print(f"    Throughput: {prefill_results['throughput']:.2f} tokens/sec")
        print(f"    Avg Latency: {prefill_results['avg_latency_ms']:.2f} ms")
    
    if decode_results:
        print(f"\n🔸 DECODE-ONLY (with LMCache SSD cache):")
        print(f"    Throughput: {decode_results['throughput']:.2f} tokens/sec")
        print(f"    Avg TPOT: {decode_results['avg_tpot_ms']:.2f} ms")
        print(f"    Avg Latency: {decode_results['avg_latency_ms']:.2f} ms")
    
    if prefill_results and decode_results:
        print(f"\n🎯 Prefill vs Decode:")
        print(f"    Prefill speed: {prefill_results['throughput']:.2f} tokens/sec")
        print(f"    Decode speed: {decode_results['throughput']:.2f} tokens/sec")
        ratio = decode_results['throughput'] / prefill_results['throughput']
        print(f"    Ratio (decode/prefill): {ratio:.2f}x")
    
    json_file = save_results(config, prefill_results, decode_results)
    
    print("\n✅ Benchmark complete with LMCache SSD storage!")
    print("="*80)

if __name__ == "__main__":
    main()

# vLLM Benchmark Refactor: Server + Sidecar Client

## Implementation Plan

---

## 1. File Inventory

### Files Modified
| File | Changes |
|------|---------|
| `automatic_launch_1.py` | Major rewrite of `generate_benchmark_script()`, `generate_yaml()` run block, `group_by_input_output_then_cluster()` regrouping logic |
| `config_generation.py` | No changes |

### New Files Created
| File | Purpose |
|------|---------|
| `benchmark_client.py` | Standalone async HTTP client with semaphore-gated concurrency. Bundled into `roofline_benchmarks/` workdir and uploaded to cluster. NOT generated as an f-string — it's a static file. |
| `prometheus_parser.py` | Parse Prometheus text exposition format, compute deltas, extract histogram percentiles. Also a static file bundled into workdir. |

### Code Deleted / Replaced
| What | Why |
|------|-----|
| `SchedulerMonitor` class (entire ~120 lines in generated script) | Replaced by `/metrics` scraping for `num_requests_running/waiting/swapped` |
| `extract_vllm_metrics()` function (~180 lines in generated script) | Replaced by `prometheus_parser.py` scraping `/metrics` endpoint |
| `get_vllm_config_info()` function (~130 lines in generated script) | Partially replaced: KV cache info from `/metrics`, server config from startup args. Some info (max_num_seqs, block_size) extracted from server startup logs or `/metrics`. |
| `LLM()` instantiation + `llm.generate()` call pattern | Replaced by `vllm serve` process + async HTTP client |
| `cleanup_dist_env_and_memory()` calls | Replaced by `kill` on server process |
| LMCache imports/config (currently commented out) | Remove entirely — not relevant to server mode |

### Code PRESERVED Unchanged
- `GPUMonitorActor` (Ray distributed) — unchanged
- `GPUMonitor` (local) — unchanged  
- `DistributedGPUMonitor` — unchanged
- `compute_canonical_columns()` — unchanged (fed from different data source)
- `get_model_config_info()` — unchanged (HuggingFace config loading)
- `get_cluster_config()` — unchanged
- `load_experiments()`, `group_by_cluster()` — unchanged
- `run_cluster_benchmarks()` — minor changes (see Section 6)
- `save_results_csv()` — unchanged
- `main()` CLI parsing — unchanged
- `send_discord_message()` — unchanged
- All SkyPilot YAML setup block (drivers, venv, vLLM install) — unchanged
- All cluster lifecycle code (sky launch/down, scp, cleanup handlers) — unchanged

---

## 2. Server Launch Mechanism

### Current YAML `run:` Block
Currently runs: `python roofline_benchmarks/benchmark_{cluster_name}.py`

### New YAML `run:` Block Structure

The generated benchmark script changes from "instantiate LLM, call generate" to a 3-phase approach:

```
Phase 1: Start vllm serve as background process
Phase 2: Wait for /health to return 200
Phase 3: Run benchmark_client.py against the server for each I/O shape
Phase 4: Kill server
```

#### Changes to `generate_yaml()`

The `run:` block's Ray setup stays the same. The change is what `benchmark_{cluster_name}.py` does internally.

#### Changes to `generate_benchmark_script()`

The generated script becomes an **orchestrator** that:

1. **Starts the vLLM server** as a subprocess:
```python
server_cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_path,
    "--tensor-parallel-size", str(tp),
    "--pipeline-parallel-size", str(pp),
    "--max-model-len", str(max_model_len),
    "--gpu-memory-utilization", "0.85",
    "--enforce-eager",
    "--disable-log-requests",
    "--port", "8000",
    # For PP > 1 with Ray:
    "--distributed-executor-backend", "ray",
]
server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
```

2. **Health check loop** (max 600s timeout for large model loading):
```python
import urllib.request
for attempt in range(1200):  # 600 seconds
    try:
        resp = urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        if resp.status == 200:
            break
    except:
        pass
    time.sleep(0.5)
else:
    raise RuntimeError("vLLM server failed to start within 600s")
```

3. **Runs benchmark_client.py** for each experiment's I/O shape (see Section 3)

4. **Kills server**: `server_proc.terminate(); server_proc.wait(timeout=30)`

#### Ray Considerations

- For PP > 1, Ray must be started BEFORE the server (current YAML already handles this)
- The server uses `--distributed-executor-backend ray` which connects to the existing Ray cluster
- GPU monitoring actors can coexist with the server since they use `num_cpus=0.1` and no GPUs

#### vLLM Version Compatibility

- vLLM 0.7.3 (A100): `python -m vllm.entrypoints.openai.api_server` (works, older entrypoint)
- vLLM 0.10.0 (L40S/L4): `vllm serve` or `python -m vllm.entrypoints.openai.api_server` (both work)
- Use `python -m vllm.entrypoints.openai.api_server` for maximum compatibility

---

## 3. Async Client Implementation (`benchmark_client.py`)

This is a **static Python file** (not generated via f-string). It's invoked by the generated orchestrator script with CLI args.

### Interface

```bash
python benchmark_client.py \
    --base-url http://localhost:8000 \
    --model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
    --input-len 8192 \
    --output-len 2048 \
    --num-requests 150 \
    --target-concurrency 50 \
    --warmup-requests 5 \
    --output /tmp/client_results_tp4_pp2_in8192_out2048.json
```

### Target Concurrency Determination

**Problem:** Currently `max_concurrency = (num_gpu_blocks * block_size) / max_model_len` is computed from `llm.llm_engine.cache_config`. With the server, we can't access this directly.

**Solution — 3 options (implement in order of preference):**

1. **Scrape `/metrics` at startup:** After server is healthy, scrape `vllm:gpu_cache_usage_perc` (should be ~0 at start). Also look for `vllm:num_gpu_blocks_total` or similar counter. If not available:

2. **Parse server startup logs:** The vLLM server prints cache config during startup, e.g.:
   ```
   INFO: # GPU blocks: 1234, # CPU blocks: 256
   ```
   Parse the server's stdout log for this line. Extract `num_gpu_blocks`, compute:
   ```
   max_concurrency = (num_gpu_blocks * block_size) / max_model_len
   target_concurrency = max_concurrency  # Use 1x, NOT 4x
   ```

3. **Fallback:** Pass `--target-concurrency` from the orchestrator based on a conservative estimate: `max(8, total_gpu_mem_gb * 1024 / (max_model_len * estimated_bytes_per_token))`.

**The orchestrator script** parses the server log to extract `num_gpu_blocks` and `block_size`, computes `target_concurrency`, and passes it to `benchmark_client.py`.

### Total Requests

```python
NUM_REQUESTS = max(50, 3 * target_concurrency)  # 3x buffer (down from 4x)
```

### Core Async Logic

```python
import asyncio
import aiohttp
import time
import json

async def run_benchmark(args):
    semaphore = asyncio.Semaphore(args.target_concurrency)
    results = []
    
    connector = aiohttp.TCPConnector(limit=args.target_concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=600)  # 10 min per request max
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        async def send_request(request_id, prompt):
            async with semaphore:
                payload = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.output_len,
                    "min_tokens": args.output_len,
                    "temperature": 0.8,
                    "ignore_eos": True,
                    "stream": True,  # For TTFT measurement
                }
                
                t_start = time.perf_counter()
                ttft = None
                chunks = []
                
                async with session.post(
                    f"{args.base_url}/v1/completions",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        return {"request_id": request_id, "status": "error", "error": error}
                    
                    # Stream SSE for TTFT
                    async for line in resp.content:
                        line = line.decode().strip()
                        if not line or not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            break
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        chunk = json.loads(line[6:])
                        chunks.append(chunk)
                
                t_end = time.perf_counter()
                e2e = t_end - t_start
                
                # Extract token counts from final chunk usage
                output_tokens = chunks[-1].get("usage", {}).get("completion_tokens", 0) if chunks else 0
                prompt_tokens = chunks[-1].get("usage", {}).get("prompt_tokens", 0) if chunks else 0
                
                return {
                    "request_id": request_id,
                    "status": "success",
                    "ttft_s": ttft,
                    "e2e_s": e2e,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                }
        
        # Fire all coroutines (semaphore gates concurrency)
        tasks = [send_request(i, prompts[i]) for i in range(len(prompts))]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

### Client-Side Metrics

The client computes and saves:
- Per-request: `ttft_s`, `e2e_s`, `prompt_tokens`, `output_tokens`
- Aggregate: percentiles (p50/p95/p99) for TTFT and e2e
- `tpot_s` per request: `(e2e_s - ttft_s) / (output_tokens - 1)` if output_tokens > 1
- Wall-clock elapsed time for the entire batch
- Total tokens processed

### Output Format

```json
{
    "config": {"input_len": 8192, "output_len": 2048, "target_concurrency": 50, "num_requests": 150},
    "wall_clock_s": 45.2,
    "total_prompt_tokens": 1228800,
    "total_output_tokens": 307200,
    "requests": [...],  // per-request details
    "client_percentiles": {
        "ttft_ms_p50": 123.4, "ttft_ms_p95": 234.5, "ttft_ms_p99": 345.6,
        "tpot_ms_p50": 12.3, "tpot_ms_p95": 23.4, "tpot_ms_p99": 34.5,
        "e2e_ms_p50": 5432.1, "e2e_ms_p95": 6543.2, "e2e_ms_p99": 7654.3
    }
}
```

---

## 4. Prometheus Metrics Scraping + Parsing (`prometheus_parser.py`)

### Scraping

```python
import urllib.request

def scrape_metrics(base_url="http://localhost:8000"):
    """Scrape /metrics endpoint, return raw text."""
    resp = urllib.request.urlopen(f"{base_url}/metrics", timeout=10)
    return resp.read().decode()
```

### Parsing Prometheus Text Format

Don't use `prometheus_client` (heavy dependency). Parse manually — the format is simple:

```
# HELP vllm:prompt_tokens_total ...
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 12345.0

# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 10
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 25
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 30
vllm:e2e_request_latency_seconds_sum 45.67
vllm:e2e_request_latency_seconds_count 30
```

Parser implementation:

```python
import re
from collections import defaultdict

def parse_prometheus_text(text):
    """Parse Prometheus text exposition format.
    
    Returns:
        {
            "counters": {"vllm:prompt_tokens_total": 12345.0, ...},
            "gauges": {"vllm:gpu_cache_usage_perc": 0.85, ...},
            "histograms": {
                "vllm:e2e_request_latency_seconds": {
                    "sum": 45.67,
                    "count": 30,
                    "buckets": [(0.5, 10), (1.0, 25), (float('inf'), 30)]
                }
            }
        }
    """
    types = {}  # metric_name -> "counter"|"gauge"|"histogram"
    result = {"counters": {}, "gauges": {}, "histograms": defaultdict(lambda: {"sum": 0, "count": 0, "buckets": []})}
    
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# TYPE "):
            parts = line.split()
            if len(parts) >= 4:
                types[parts[2]] = parts[3]
            continue
        if line.startswith("#") or not line:
            continue
        
        # Parse metric line: name{labels} value
        match = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*?)(?:_bucket\{le="([^"]+)"\}|_sum|_count|(?:\{[^}]*\})?)?\s+(\S+)', line)
        if not match:
            continue
        
        # ... (full parsing logic for buckets, sum, count, plain values)
    
    return result
```

### Metric Name Mapping

| Prometheus Metric | Maps To | Type |
|---|---|---|
| `vllm:prompt_tokens_total` | `prompt_tokens_total` | counter |
| `vllm:generation_tokens_total` | `generation_tokens_total` | counter |
| `vllm:e2e_request_latency_seconds` | e2e percentiles + avg | histogram |
| `vllm:time_to_first_token_seconds` | TTFT percentiles + avg | histogram |
| `vllm:time_per_output_token_seconds` | TPOT percentiles + avg | histogram |
| `vllm:request_prefill_time_seconds` | prefill time avg | histogram |
| `vllm:request_decode_time_seconds` | decode time avg | histogram |
| `vllm:request_prompt_tokens` | avg prompt len | histogram |
| `vllm:request_generation_tokens` | avg gen len | histogram |
| `vllm:gpu_cache_usage_perc` | KV cache utilization (for timeseries) | gauge |
| `vllm:cpu_cache_usage_perc` | CPU cache util | gauge |
| `vllm:num_requests_running` | scheduler running queue | gauge |
| `vllm:num_requests_waiting` | scheduler waiting queue | gauge |
| `vllm:num_requests_swapped` | scheduler swapped queue | gauge |
| `vllm:num_preemptions_total` | **NEW** preemption count | counter |

### Percentile Computation from Histogram Buckets

Reuse the existing `_histogram_quantile` logic (already in `extract_vllm_metrics`). Move it to `prometheus_parser.py` as a shared utility.

---

## 5. Warmup + Delta Metrics Flow

### Step-by-Step Flow (in the generated orchestrator script)

```
1. Start vLLM server → wait for /health 200
2. Parse server logs → extract num_gpu_blocks, block_size → compute target_concurrency
3. Start GPU monitoring (same as today)
4. Run warmup: benchmark_client.py --num-requests 5 --target-concurrency 5 [discard results]
5. SCRAPE /metrics → save as warmup_metrics (prometheus_parser.parse_prometheus_text)
6. Start scheduler metrics timeseries polling (see below)
7. Run real benchmark: benchmark_client.py --num-requests N --target-concurrency K
8. SCRAPE /metrics → save as final_metrics
9. Stop GPU monitoring, stop scheduler polling
10. Compute deltas: final_metrics - warmup_metrics for all counters and histograms
11. Build canonical columns from deltas + client results + GPU metrics
12. Save results.json + timeseries_*.json
```

### Scheduler Timeseries Replacement

Instead of `SchedulerMonitor` polling engine internals, poll `/metrics` every 0.25s during the benchmark run:

```python
class MetricsPoller:
    """Poll /metrics endpoint periodically for timeseries."""
    
    def __init__(self, base_url="http://localhost:8000", interval=0.5):
        self.base_url = base_url
        self.interval = interval
        self.timeseries = []
        self._stop = threading.Event()
        self._thread = None
    
    def start(self):
        self._stop.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def _loop(self):
        while not self._stop.is_set():
            try:
                text = scrape_metrics(self.base_url)
                parsed = parse_prometheus_text(text)
                sample = {
                    "t": round(time.time() - self._start_time, 3),
                    "running": parsed["gauges"].get("vllm:num_requests_running", 0),
                    "waiting": parsed["gauges"].get("vllm:num_requests_waiting", 0),
                    "swapped": parsed["gauges"].get("vllm:num_requests_swapped", 0),
                    "kv_cache_util_pct": round(parsed["gauges"].get("vllm:gpu_cache_usage_perc", 0) * 100, 1),
                }
                self.timeseries.append(sample)
            except:
                pass
            self._stop.wait(self.interval)
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
```

This replaces `SchedulerMonitor` entirely and produces the same timeseries format.

### Delta Computation

```python
def compute_deltas(warmup_parsed, final_parsed):
    """Compute metric deltas between two /metrics scrapes."""
    deltas = {}
    
    # Counter deltas
    for name in final_parsed["counters"]:
        deltas[name] = final_parsed["counters"][name] - warmup_parsed["counters"].get(name, 0)
    
    # Histogram deltas (sum, count, bucket counts)
    for name in final_parsed["histograms"]:
        wh = warmup_parsed["histograms"].get(name, {"sum": 0, "count": 0, "buckets": []})
        fh = final_parsed["histograms"][name]
        
        delta_sum = fh["sum"] - wh["sum"]
        delta_count = fh["count"] - wh["count"]
        
        # Bucket delta: subtract warmup cumulative counts from final
        wb_dict = {b: c for b, c in wh["buckets"]}
        delta_buckets = [(b, c - wb_dict.get(b, 0)) for b, c in fh["buckets"]]
        
        deltas[name] = {
            "sum": delta_sum,
            "count": delta_count,
            "avg": delta_sum / delta_count if delta_count > 0 else None,
            "buckets": delta_buckets,
        }
    
    # Gauges: just use final values (they're point-in-time)
    deltas["gauges"] = final_parsed["gauges"]
    
    return deltas
```

---

## 6. Server Reuse Across I/O Shapes

### Current Grouping

`group_by_input_output_then_cluster()` groups by `(input_len, output_len)` → `(gpus_per_node, num_nodes)`.

Each unique `(gpus_per_node, num_nodes, input_len, output_len)` combo gets its own cluster + benchmark script. The model is loaded fresh for every single benchmark run.

### New Grouping: Cluster-First

**Key insight:** With the server architecture, we can start the server ONCE per (TP, PP, model) and run multiple I/O shapes against it. The server's `max_model_len` must be >= the largest `input_len + output_len` in the group.

#### New Grouping Function

```python
def group_by_cluster_then_io(experiments, gpu_type, vm_strategy):
    """Group experiments: same cluster config runs multiple I/O shapes.
    
    Returns: {(gpus_per_node, num_nodes): [experiments]}
    
    Within each cluster group, experiments may have different I/O shapes.
    The server is started with max_model_len = max(input+output) across all exps.
    """
    groups = defaultdict(list)
    for exp in experiments:
        gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'], gpu_type, vm_strategy)
        # Group by (gpus_per_node, num_nodes, tp, pp, model) — same server config
        key = (gpus_per_node, num_nodes, exp['tp'], exp['pp'], exp['model'])
        groups[key].append(exp)
    return groups
```

#### How the Generated Script Changes

The generated orchestrator script receives ALL experiments for a given (TP, PP, model) combo:

```python
EXPERIMENTS = [
    {"tp": 4, "pp": 2, "max_input_length": 4096, "max_output_length": 1024, "model": "..."},
    {"tp": 4, "pp": 2, "max_input_length": 8192, "max_output_length": 2048, "model": "..."},
    {"tp": 4, "pp": 2, "max_input_length": 30000, "max_output_length": 7000, "model": "..."},
]

# Start server with max_model_len covering ALL experiments
MAX_MODEL_LEN = max(e['max_input_length'] + e['max_output_length'] for e in EXPERIMENTS)
# ... start server once ...

# Run each I/O shape against the same server
for exp in EXPERIMENTS:
    # Run warmup → scrape → benchmark → scrape → delta for this I/O shape
    result = run_single_benchmark(exp, server_url="http://localhost:8000")
    results.append(result)

# Kill server once after all experiments
```

#### Impact on `run_cluster_benchmarks()`

Minor changes:
- Cluster naming: use `roofline-tp{tp}-pp{pp}-{gpu_type}` (no I/O in name since one cluster serves multiple I/O shapes)
- Result directory: `tp{tp}-pp{pp}-{instance}-{timestamp}` (contains results for all I/O shapes)
- The YAML and script generation receives the full experiment list for that cluster

#### Impact on `main()`

The outer loop changes from iterating `io_groups[io_key][cluster_key]` to iterating `cluster_groups[cluster_key]` directly:

```python
cluster_groups = group_by_cluster_then_io(experiments, gpu_type, vm_strategy)
for (gpus, nodes, tp, pp, model), exps in sorted(cluster_groups.items()):
    results = run_cluster_benchmarks((gpus, nodes), exps, ...)
```

#### Savings Estimate

For a typical config with 3 I/O shapes × 15 TP/PP combos = 45 experiments:
- **Before:** 45 cluster launches, 45 model loads
- **After:** 15 cluster launches, 15 model loads, 45 benchmark runs
- **Estimated time savings:** ~60% (model loading is typically 2-5 min per load)

---

## 7. Error Handling

### Server Fails to Start (OOM during model load)

```python
# In orchestrator:
server_proc = subprocess.Popen(server_cmd, ...)

# Health check with timeout
for attempt in range(1200):
    if server_proc.poll() is not None:
        # Server process died
        exit_code = server_proc.returncode
        log_content = server_log.read()
        if "OutOfMemoryError" in log_content or "CUDA out of memory" in log_content:
            error = f"OOM: Model too large for {gpu_count}x {GPU_MODEL}"
        else:
            error = f"Server crashed with exit code {exit_code}"
        # Mark ALL experiments as failed, save results, continue
        for exp in EXPERIMENTS:
            results.append({**exp, "status": "error", "error": error})
        break
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        break  # Server ready
    except:
        time.sleep(0.5)
```

### Server Crashes Mid-Benchmark

```python
# In benchmark_client.py — handle connection errors gracefully
async def send_request(request_id, prompt):
    try:
        async with semaphore:
            async with session.post(...) as resp:
                ...
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {"request_id": request_id, "status": "error", "error": str(e)}

# In orchestrator — check server health after client returns
if server_proc.poll() is not None:
    print("⚠️ Server died during benchmark")
    # Save partial results for completed requests
    successful = [r for r in client_results["requests"] if r["status"] == "success"]
    if len(successful) >= 10:
        # Enough data to compute meaningful metrics from partial run
        result["status"] = "partial"
        result["note"] = f"Server crashed, {len(successful)}/{total} requests completed"
    else:
        result["status"] = "error"
        result["error"] = "Server crashed mid-benchmark"
```

### Client Timeout

```python
# In benchmark_client.py
# Individual request timeout: 10 minutes (for very long sequences)
# Overall benchmark timeout: computed from expected throughput
INDIVIDUAL_TIMEOUT = 600  # seconds
OVERALL_TIMEOUT = max(1800, NUM_REQUESTS * 30)  # at least 30s per request

try:
    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=OVERALL_TIMEOUT
    )
except asyncio.TimeoutError:
    # Cancel remaining tasks, save what we have
    ...
```

### Partial Results Recovery

The orchestrator saves results incrementally (same as current code):
```python
for exp in EXPERIMENTS:
    result = run_single_benchmark(exp, ...)
    results.append(result)
    # Save after each experiment
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
```

If the server dies between experiments, previous experiments' results are preserved.

---

## 8. Migration Checklist

### Phase 1: Create New Files (No Breaking Changes)

- [ ] Create `prometheus_parser.py` with `parse_prometheus_text()`, `histogram_quantile()`, `compute_deltas()`
- [ ] Create `benchmark_client.py` with async HTTP client, semaphore pattern, CLI interface
- [ ] Unit test both files locally (mock /metrics responses, mock completions API)

### Phase 2: Modify Generated Script Template

- [ ] Modify `generate_benchmark_script()` to produce server-mode orchestrator
- [ ] The orchestrator imports from `prometheus_parser.py` and invokes `benchmark_client.py`
- [ ] Keep `compute_canonical_columns()` exactly the same — just change what feeds into `measured_data`
- [ ] Keep `get_model_config_info()` exactly the same
- [ ] Keep GPU monitoring code exactly the same
- [ ] Replace `SchedulerMonitor` with `MetricsPoller`

### Phase 3: Modify YAML Generation

- [ ] Modify `generate_yaml()` to include `benchmark_client.py` and `prometheus_parser.py` in workdir
- [ ] Ensure `aiohttp` is added to `uv pip install` in setup block
- [ ] No changes to Ray startup logic in `run:` block (server uses `--distributed-executor-backend ray`)

### Phase 4: Modify Grouping + Orchestration

- [ ] Add `group_by_cluster_then_io()` function
- [ ] Modify `main()` to use new grouping
- [ ] Modify `run_cluster_benchmarks()` for cluster naming without I/O in name
- [ ] Test with dry run to verify grouping

### Phase 5: Delete Dead Code

- [ ] Remove `SchedulerMonitor` class from generated script template
- [ ] Remove `extract_vllm_metrics()` from generated script template
- [ ] Remove `get_vllm_config_info()` from generated script template (keep `get_model_config_info()`)
- [ ] Remove all LMCache imports/config from generated script template
- [ ] Remove `group_by_input_output_then_cluster()` (replaced by `group_by_cluster_then_io()`)

### Phase 6: Validation

- [ ] Verify all 73 canonical columns are populated (diff CSV column headers before/after)
- [ ] Verify results.json schema is compatible
- [ ] Verify timeseries_*.json schema is compatible
- [ ] Run on L40S (vLLM 0.10.0) — primary target
- [ ] Run on A100 (vLLM 0.7.3) — verify `/metrics` endpoint exists in 0.7.3
- [ ] Compare throughput numbers old vs new (expect new to be HIGHER due to no thrashing)
- [ ] Verify preemption count from `/metrics` is ~0 (confirms no thrashing)

### ⚠️ Risk: vLLM 0.7.3 `/metrics` Availability

**CRITICAL CHECK:** Verify that vLLM 0.7.3's OpenAI server exposes `/metrics`. The Prometheus metrics endpoint was added relatively early, but the specific metric names may differ.

```bash
# Test locally or on a cluster:
pip install vllm==0.7.3
python -m vllm.entrypoints.openai.api_server --model <small-model> --port 8000 &
curl http://localhost:8000/metrics
```

If 0.7.3 doesn't have `/metrics`, options:
1. Fall back to in-process `LLM()` mode for A100 (maintain both code paths — not ideal)
2. Upgrade A100 to newer vLLM by updating drivers (preferred long-term)
3. Use client-side metrics only (lose server-side histogram precision)

---

## 9. Canonical Column Population Mapping

How each column in `compute_canonical_columns()` gets populated in the new architecture:

| Column | Current Source | New Source |
|--------|---------------|------------|
| `tokens_per_sec_total` | `(prompt_toks + gen_toks) / elapsed` | **vLLM server-side token counts** from delta of `vllm:prompt_tokens_total` + `vllm:generation_tokens_total` counters (post-run scrape minus warmup scrape), divided by wall-clock elapsed. NOT client-side counts. |
| `tokens_per_sec_prefill` | `prompt_toks / elapsed` | Delta of `vllm:prompt_tokens_total` / wall_clock_elapsed |
| `tokens_per_sec_decode` | `gen_toks / elapsed` | Delta of `vllm:generation_tokens_total` / wall_clock_elapsed |
| `total_cost_usd` | `price * elapsed / 3600` | Same |
| `max_num_seqs` | `vllm_config['max_num_seqs']` from engine internals | Parse from server startup log or pass as CLI arg |
| `batch_size` | Same as max_num_seqs | Same |
| `ttft_ms_p50/p95/p99` | Delta histogram buckets from `llm.get_metrics()` | Delta histogram buckets from `/metrics` scrape |
| `tpot_ms_p50/p95/p99` | Same | Same |
| `e2e_ms_p50/p95/p99` | Same | Same |
| `num_requests` | `NUM_SAMPLES` | `NUM_REQUESTS` from client |
| All GPU/model/config columns | Unchanged | Unchanged |

**CRITICAL: Token counts for throughput MUST come from vLLM server-side counters** (`vllm:prompt_tokens_total`, `vllm:generation_tokens_total` delta), NOT from client-side response parsing. Client-side counts may differ due to HTTP overhead, streaming chunking, or partial responses. The server knows exactly how many tokens it processed.

The `measured_data` dict passed to `compute_canonical_columns()` has the **exact same keys** — only the data source changes from engine internals to HTTP API.

---

## 10. Dependencies Added

| Package | Where | Why |
|---------|-------|-----|
| `aiohttp` | Cluster setup (`uv pip install`) | Async HTTP client for benchmark |

No other new dependencies. `asyncio` is stdlib. `urllib.request` is stdlib (used for health check and metrics scraping in orchestrator).

---

## 11. File Structure After Refactor

```
roofline/vllm/
├── automatic_launch_1.py          # Modified: new generate_benchmark_script(), grouping
├── config_generation.py           # Unchanged
├── benchmark_client.py            # NEW: async HTTP benchmark client
├── prometheus_parser.py           # NEW: /metrics parsing + delta computation
├── plot_timeseries.py             # Unchanged
├── upload_model_to_s3.py          # Unchanged
├── roofline_benchmarks/           # Generated at runtime
│   ├── benchmark_client.py        # Copied from parent (included in workdir)
│   ├── prometheus_parser.py       # Copied from parent (included in workdir)
│   ├── roofline-tp4-pp2-L40S.yaml # Generated (no I/O in name)
│   └── benchmark_roofline-tp4-pp2-L40S.py  # Generated orchestrator
└── results/
    └── wrk-8192in_2048out/        # Result directories (unchanged structure)
```

Note: `benchmark_client.py` and `prometheus_parser.py` must be copied into `roofline_benchmarks/` before `sky launch` so they're uploaded as part of the workdir. Add this to `run_cluster_benchmarks()`:

```python
# Copy static helper files into workdir
import shutil
shutil.copy2("benchmark_client.py", work_dir / "benchmark_client.py")
shutil.copy2("prometheus_parser.py", work_dir / "prometheus_parser.py")
```

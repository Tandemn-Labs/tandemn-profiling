#!/usr/bin/env python3
"""
SAM2 Production Load Generator - Async version with true concurrency
"""

import io
import time
import json
import base64
import random
import asyncio
import aiohttp
import argparse
import signal
import sys
import subprocess
import numpy as np
from PIL import Image
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from collections import deque

# =============================================================================
# CONFIG
# =============================================================================

SLO_TARGET_MS = 200
COLD_START_THRESHOLD_MS = 5000
IDLE_THRESHOLD_SECONDS = 300

# Prompts for 01.jpeg
PROMPTS = [
    {"point": [1050, 420], "label": 1},
    {"point": [1000, 380], "label": 1},
    {"point": [600, 500], "label": 1},
    {"point": [900, 550], "label": 1},
    {"point": [700, 400], "label": 1},
    {"point": [200, 300], "label": 1},
    {"point": [1400, 250], "label": 1},
    {"point": [700, 100], "label": 1},
]

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Result:
    timestamp: float
    success: bool
    latency_ms: float
    num_prompts: int
    slo_met: bool
    is_cold_start: bool
    phase: str
    error: Optional[str] = None


@dataclass
class PhaseStats:
    name: str
    duration_s: float
    requests: int
    success: int
    failed: int
    slo_violations: int
    cold_starts: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    actual_qps: float


# =============================================================================
# LOAD GENERATOR
# =============================================================================

class LoadGenerator:
    def __init__(self, endpoint: str, image_path: str, stats_interval: int = 60):
        self.endpoint = endpoint.rstrip('/')
        self.image_b64 = self._encode_image(image_path)
        self.stats_interval = stats_interval
        self.service_name = "sam2-serve"  # Default, can be overridden
        
        # State
        self.running = True
        self.results: deque = deque(maxlen=100000)  # Bounded buffer
        self.phase_stats: List[PhaseStats] = []
        self.current_phase = "init"
        self.current_qps = 0.0
        self.last_request_time = 0.0
        self.start_time = 0.0
        
        # Counters (atomic-ish for async)
        self.total_requests = 0
        self.total_success = 0
        self.total_failed = 0
        self.slo_violations = 0
        self.cold_starts = 0
        
        print(f"Loaded image ({len(self.image_b64)} bytes b64)")

    def _encode_image(self, path: str) -> str:
        with Image.open(path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()

    def _get_prompts(self, n: int) -> List[Dict]:
        n = min(n, len(PROMPTS))
        return random.sample(PROMPTS, n)

    async def _send_request(self, session: aiohttp.ClientSession, num_prompts: int) -> Result:
        """Send single request - non-blocking."""
        start = time.time()
        idle_time = start - self.last_request_time if self.last_request_time else float('inf')
        
        result = Result(
            timestamp=start,
            success=False,
            latency_ms=0,
            num_prompts=num_prompts,
            slo_met=False,
            is_cold_start=False,
            phase=self.current_phase,
        )
        
        try:
            payload = {
                "image": self.image_b64,
                "prompts": self._get_prompts(num_prompts),
                "multimask_output": False,
            }
            
            async with session.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                latency = (time.time() - start) * 1000
                result.latency_ms = latency
                
                if resp.status == 200:
                    result.success = True
                    result.slo_met = latency <= SLO_TARGET_MS
                    
                    # Cold start detection
                    if idle_time > IDLE_THRESHOLD_SECONDS and latency > COLD_START_THRESHOLD_MS:
                        result.is_cold_start = True
                else:
                    result.error = f"HTTP {resp.status}"
                    
        except asyncio.TimeoutError:
            result.latency_ms = (time.time() - start) * 1000
            result.error = "timeout"
            if idle_time > IDLE_THRESHOLD_SECONDS:
                result.is_cold_start = True
        except Exception as e:
            result.latency_ms = (time.time() - start) * 1000
            result.error = str(e)[:50]
        
        # Update counters
        self.last_request_time = time.time()
        self.total_requests += 1
        if result.success:
            self.total_success += 1
            if not result.slo_met:
                self.slo_violations += 1
        else:
            self.total_failed += 1
        if result.is_cold_start:
            self.cold_starts += 1
        
        self.results.append(result)
        return result

    async def _generate_load(
        self,
        session: aiohttp.ClientSession,
        qps: float,
        prompts_range: Tuple[int, int],
        duration_s: float,
    ):
        """Generate continuous load at target QPS using token bucket."""
        if qps <= 0:
            return
        
        interval = 1.0 / qps
        phase_start = time.time()
        next_send = phase_start
        pending = set()
        
        while self.running and (time.time() - phase_start) < duration_s:
            now = time.time()
            
            # Send requests up to current time
            while next_send <= now and self.running:
                num_prompts = random.randint(prompts_range[0], prompts_range[1])
                task = asyncio.create_task(self._send_request(session, num_prompts))
                pending.add(task)
                task.add_done_callback(pending.discard)
                next_send += interval
            
            # Clean up completed tasks periodically
            if len(pending) > 100:
                done, pending = await asyncio.wait(
                    pending, timeout=0.01, return_when=asyncio.FIRST_COMPLETED
                )
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.001)
        
        # Wait for remaining requests
        if pending:
            await asyncio.wait(pending, timeout=30)

    async def _run_phase(
        self,
        session: aiohttp.ClientSession,
        name: str,
        duration_min: float,
        qps_range: Tuple[float, float],
        prompts_range: Tuple[int, int],
        idle: bool = False,
    ) -> PhaseStats:
        """Run single test phase."""
        print(f"\n{'='*50}")
        print(f"Phase: {name}")
        print(f"Duration: {duration_min:.1f}min | QPS: {qps_range} | Prompts: {prompts_range}")
        print(f"{'='*50}")
        
        self.current_phase = name
        phase_start = time.time()
        duration_s = duration_min * 60
        start_count = self.total_requests
        
        if idle:
            print("Idle phase - no requests")
            await asyncio.sleep(duration_s)
        else:
            # Variable QPS - change every 30-60s
            elapsed = 0
            while elapsed < duration_s and self.running:
                qps = random.uniform(qps_range[0], qps_range[1])
                self.current_qps = qps
                chunk_duration = min(random.uniform(30, 60), duration_s - elapsed)
                
                print(f"  QPS: {qps:.1f} for {chunk_duration:.0f}s")
                await self._generate_load(session, qps, prompts_range, chunk_duration)
                elapsed = time.time() - phase_start
        
        # Compute phase stats
        phase_duration = time.time() - phase_start
        phase_results = [r for r in self.results if r.phase == name]
        successful = [r for r in phase_results if r.success]
        latencies = [r.latency_ms for r in successful if not r.is_cold_start]
        
        stats = PhaseStats(
            name=name,
            duration_s=phase_duration,
            requests=len(phase_results),
            success=len(successful),
            failed=len(phase_results) - len(successful),
            slo_violations=sum(1 for r in successful if not r.slo_met),
            cold_starts=sum(1 for r in phase_results if r.is_cold_start),
            avg_ms=np.mean(latencies) if latencies else 0,
            p50_ms=np.percentile(latencies, 50) if latencies else 0,
            p95_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_ms=np.percentile(latencies, 99) if latencies else 0,
            actual_qps=len(phase_results) / phase_duration if phase_duration > 0 else 0,
        )
        
        self.phase_stats.append(stats)
        print(f"  Done: {stats.requests} req, {stats.actual_qps:.1f} QPS, P95={stats.p95_ms:.0f}ms")
        return stats

    async def _stats_logger(self):
        """Background task for periodic stats."""
        while self.running:
            await asyncio.sleep(self.stats_interval)
            if not self.running:
                break
            self._print_stats()

    def _print_stats(self):
        """Print current stats."""
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60
        
        # Get recent latencies
        recent = [r for r in self.results if r.success and not r.is_cold_start][-1000:]
        latencies = [r.latency_ms for r in recent]
        
        print(f"\n{'─'*60}")
        print(f"STATS @ {datetime.now().strftime('%H:%M:%S')} | Elapsed: {elapsed_min:.1f}min")
        print(f"{'─'*60}")
        print(f"Requests: {self.total_requests:,} | Success: {self.total_success:,} ({100*self.total_success/max(1,self.total_requests):.1f}%)")
        print(f"SLO Violations: {self.slo_violations:,} | Cold Starts: {self.cold_starts}")
        print(f"QPS: {self.total_requests/elapsed:.1f} actual | Phase: {self.current_phase}")
        if latencies:
            print(f"Latency: Avg={np.mean(latencies):.0f}ms P50={np.percentile(latencies,50):.0f}ms P95={np.percentile(latencies,95):.0f}ms P99={np.percentile(latencies,99):.0f}ms")
        print(f"{'─'*60}")

    def _generate_report(self) -> Dict:
        """Generate final report."""
        all_results = list(self.results)
        successful = [r for r in all_results if r.success]
        warm = [r for r in successful if not r.is_cold_start]
        warm_latencies = [r.latency_ms for r in warm]
        cold_latencies = [r.latency_ms for r in all_results if r.is_cold_start and r.success]
        
        report = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_hours": (time.time() - self.start_time) / 3600,
            "total_requests": self.total_requests,
            "successful": self.total_success,
            "failed": self.total_failed,
            "success_rate": self.total_success / max(1, self.total_requests),
            "slo_target_ms": SLO_TARGET_MS,
            "slo_violations": self.slo_violations,
            "slo_violation_rate": self.slo_violations / max(1, self.total_success),
            "cold_starts": self.cold_starts,
            "avg_cold_start_ms": np.mean(cold_latencies) if cold_latencies else 0,
            "avg_warm_latency_ms": np.mean(warm_latencies) if warm_latencies else 0,
            "p50_warm_latency_ms": np.percentile(warm_latencies, 50) if warm_latencies else 0,
            "p95_warm_latency_ms": np.percentile(warm_latencies, 95) if warm_latencies else 0,
            "p99_warm_latency_ms": np.percentile(warm_latencies, 99) if warm_latencies else 0,
            "phases": [asdict(p) for p in self.phase_stats],
        }
        return report

    def _print_summary(self, report: Dict):
        """Print final summary."""
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Duration: {report['duration_hours']:.2f} hours")
        print(f"Total Requests: {report['total_requests']:,}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"SLO Violations: {report['slo_violations']:,} ({report['slo_violation_rate']*100:.2f}%)")
        print(f"Cold Starts: {report['cold_starts']}")
        print(f"\nWarm Latency:")
        print(f"  Avg: {report['avg_warm_latency_ms']:.1f}ms")
        print(f"  P50: {report['p50_warm_latency_ms']:.1f}ms")
        print(f"  P95: {report['p95_warm_latency_ms']:.1f}ms")
        print(f"  P99: {report['p99_warm_latency_ms']:.1f}ms")
        if report['cold_starts'] > 0:
            print(f"  Cold Start Avg: {report['avg_cold_start_ms']:.0f}ms")
        print("="*60)

    def _save_autoscaler_logs(self):
        """Save autoscaler logs to file."""
        print(f"\nSaving autoscaler logs for service '{self.service_name}'...")
        try:
            # Run sky serve logs command with grep filters
            cmd = f'sky serve logs {self.service_name} --controller | grep -E "scale|replicas" | grep -E "Target|Scaling"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and result.stdout.strip():
                filename = "autoscaler_logs.txt"
                with open(filename, 'w') as f:
                    f.write(result.stdout)
                print(f"Autoscaler logs saved to: {filename}")
                
                # Count lines
                line_count = len(result.stdout.strip().split('\n'))
                print(f"  Captured {line_count} log entries")
            else:
                print("No autoscaler logs found or service not available")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("Warning: Timeout while fetching autoscaler logs")
        except Exception as e:
            print(f"Warning: Could not save autoscaler logs: {e}")

    async def run_benchmark(self, duration_hours: float, quick: bool = False):
        """Run the full benchmark."""
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("SAM2 LOAD GENERATOR")
        print("="*60)
        print(f"Endpoint: {self.endpoint}")
        print(f"Duration: {duration_hours}h | SLO: {SLO_TARGET_MS}ms")
        print("="*60)
        
        # Define phases
        if quick:
            phases = [
                ("Warmup", 1, (3, 5), (3, 5), False),
                ("Low", 2, (5, 10), (5, 10), False),
                ("Medium", 3, (10, 15), (5, 15), False),
                ("High", 3, (15, 25), (10, 20), False),
                ("Idle", 5, (0, 0), (0, 0), True),
                ("ColdStart", 2, (10, 15), (5, 15), False),
                ("Burst", 2, (20, 25), (15, 25), False),
                ("Cooldown", 1, (5, 5), (5, 5), False),
            ]
        else:
            phases = [
                ("Warmup", 5, (5, 5), (1, 5), False),
                ("Ramp", 30, (5, 15), (2, 8), False),
                ("Peak1", 60, (15, 25), (5, 15), False),
                ("Sustained", 90, (10, 20), (2, 8), False),
                ("Idle1", 15, (0, 0), (0, 0), True),
                ("ColdStart1", 10, (5, 10), (1, 5), False),
                ("Midday", 120, (8, 18), (2, 8), False),
                ("Spike", 30, (20, 25), (5, 15), False),
                ("Idle2", 20, (0, 0), (0, 0), True),
                ("ColdBurst", 5, (15, 20), (10, 20), False),
                ("Afternoon", 120, (10, 15), (8, 18), False),
                ("Idle3", 25, (0, 0), (0, 0), True),
                ("Evening", 150, (8, 18), (2, 8), False),
                ("Night", 120, (5, 10), (1,5), False),
                ("OvernightIdle", 60, (0, 0), (0, 0), True),
                ("EarlyMorning", 60, (5, 15), (2, 8), False),
                ("FinalBurst", 30, (20, 25), (20, 25), False),
                ("Cooldown", 10, (5, 5), (5, 5), False),
            ]
        
        # Scale phases to requested duration
        total_min = sum(p[1] for p in phases)
        target_min = duration_hours * 60
        scale = target_min / total_min
        
        print(f"Phases: {len(phases)} | Scale: {scale:.2f}x")
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=50, keepalive_timeout=30)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Start stats logger
            stats_task = asyncio.create_task(self._stats_logger())
            
            try:
                for name, dur, qps, prompts, idle in phases:
                    if not self.running:
                        break
                    await self._run_phase(
                        session, name, dur * scale, qps, prompts, idle
                    )
            finally:
                stats_task.cancel()
                try:
                    await stats_task
                except asyncio.CancelledError:
                    pass
        
        # Generate and save report
        report = self._generate_report()
        self._print_summary(report)
        
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved to: {filename}")
        
        # Save autoscaler logs
        self._save_autoscaler_logs()
        
        return report

    def stop(self):
        print("\nStopping...")
        self.running = False


# =============================================================================
# MAIN
# =============================================================================

async def async_main():
    parser = argparse.ArgumentParser(description='SAM2 Production Load Generator')
    parser.add_argument('--endpoint', required=True, help='SAM2 endpoint URL')
    parser.add_argument('--image', default='01.jpeg', help='Test image path')
    parser.add_argument('--hours', type=float, default=24, help='Duration in hours')
    parser.add_argument('--quick', action='store_true', help='Quick test (~20min)')
    parser.add_argument('--quick-minutes', type=int, default=20, help='Quick test duration')
    parser.add_argument('--stats-interval', type=int, default=60, help='Stats log interval (s)')
    parser.add_argument('--service-name', default='sam2-serve', help='SkyServe service name for log collection')
    args = parser.parse_args()
    
    gen = LoadGenerator(args.endpoint, args.image, args.stats_interval)
    gen.service_name = args.service_name
    
    # Signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, gen.stop)
    
    try:
        hours = args.quick_minutes / 60 if args.quick else args.hours
        await gen.run_benchmark(hours, args.quick)
        print("\nBenchmark complete!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())

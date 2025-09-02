#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prefill/Refill the 'hard' Sudoku cache file (JSON list) as fast as possible.

- Uses multiprocessing for maximum throughput (CPU-bound).
- Prints a short line for each completed puzzle.
- Deduplicates by solution grid to keep variety.
- Safe atomic writes to cache file.
- Parameters allow target size, max file size, worker count, etc.

Usage examples:
  python prefill_hard_cache.py                         # default target = 100, workers = CPU cores
  python prefill_hard_cache.py --target 200 --workers 12
  python prefill_hard_cache.py --outfile hard_cache.json --max-file 300

Requires:
  - Sudoko.py in same directory (must expose generate_puzzle)
  - numpy (imported by Sudoko)
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import itertools
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import generation from your game module
try:
    from Sudoko import generate_puzzle
except Exception as e:
    print("ERROR: Could not import generate_puzzle from Sudoko.py")
    print("Make sure prefill_hard_cache.py is in the same directory as Sudoko.py")
    raise

# Try to pick a default outfile from your module; fall back to hard_cache.json
DEFAULT_OUTFILE = "hard_cache.json"
try:
    from Sudoko import HARD_CACHE_FILE
    DEFAULT_OUTFILE = HARD_CACHE_FILE
except Exception:
    pass


def safe_atomic_write_json(path: str, data: Any) -> None:
    """Write JSON safely by using a temporary file and os.replace()."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def flatten_grid(grid: List[List[int]]) -> List[int]:
    return [v for row in grid for v in row]


def grid_hash(grid: List[List[int]]) -> str:
    # Fast, simple hash for 9x9 integer grids
    return "".join(str(v) for v in flatten_grid(grid))


def load_existing_cache(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def worker_task(seed: Optional[int] = None) -> Dict[str, Any]:
    # Optional: set per-process seed for variety
    if seed is not None:
        import random
        random.seed(seed)

    # Generate a hard puzzle using your module's logic (unique + logic-solvable)
    puzzle, solution = generate_puzzle("hard")
    return {"puzzle": puzzle, "solution": solution}


def main():
    parser = argparse.ArgumentParser(description="Refill hard Sudoku cache as fast as possible.")
    parser.add_argument("--outfile", type=str, default=DEFAULT_OUTFILE,
                        help=f"Path to cache JSON (default: {DEFAULT_OUTFILE})")
    parser.add_argument("--target", type=int, default=100,
                        help="Target number of puzzles in the cache file (default: 100)")
    parser.add_argument("--max-file", type=int, default=None,
                        help="Maximum entries to keep in the file (default: same as --target)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of worker processes (default: CPU cores)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Write-to-disk batch size (default: 8)")
    parser.add_argument("--dedup", action="store_true",
                        help="Enable deduplication by solution (recommended)")
    args = parser.parse_args()

    outfile = args.outfile
    target = int(args.target)
    max_file = int(args.max_file) if args.max_file else target
    workers = max(1, int(args.workers))
    batch_size = max(1, int(args.batch))
    dedup = bool(args.dedup)

    print(f"Prefill hard cache -> {outfile}")
    print(f"Target size: {target} | Max file size: {max_file} | Workers: {workers} | Batch write: {batch_size} | Dedup: {dedup}")
    start_all = time.time()

    # Load existing file
    cache_list = load_existing_cache(outfile)
    print(f"Loaded existing cache entries: {len(cache_list)}")

    # Build a set of solution hashes for dedup (if requested)
    seen_hashes = set()
    if dedup:
        for item in cache_list:
            try:
                h = grid_hash(item["solution"])
                seen_hashes.add(h)
            except Exception:
                continue

    # If we already have >= target, trim and exit quickly
    if len(cache_list) >= target:
        cache_list = cache_list[:max_file]
        safe_atomic_write_json(outfile, cache_list)
        print(f"Cache already at/above target. Trimmed to {len(cache_list)}. Done.")
        return

    # Number we still need
    need = target - len(cache_list)
    print(f"Need to generate: {need}")

    # Launch process pool
    # Use a stream of seeds for variety (optional)
    seeds = itertools.count(1)  # 1,2,3,... per task
    futures = []
    produced = 0
    last_write = time.time()

    # Submit work
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i in range(need):
            futures.append(pool.submit(worker_task, next(seeds)))

        # Consume as completed; keep writing to disk in small batches
        pending = len(futures)
        batch_buffer: List[Dict[str, Any]] = []
        i_done = 0

        for fut in as_completed(futures):
            t0 = time.time()
            try:
                result = fut.result()
                puzzle = result["puzzle"]
                solution = result["solution"]
                ok = True

                if dedup:
                    h = grid_hash(solution)
                    if h in seen_hashes:
                        ok = False
                    else:
                        seen_hashes.add(h)

                if ok:
                    # Prepend newest (so consumers read newest first)
                    cache_list.insert(0, {"puzzle": puzzle, "solution": solution})
                    produced += 1
                    # Short per-item output
                    dt = time.time() - t0
                    print(f"[{produced}/{need}] cached one hard puzzle (+1) in {dt:.2f}s | total file: {len(cache_list)}")
                else:
                    # Duplicate—skip counting towards 'produced', but still show minimal feedback
                    print(f"[dup] skipped duplicate solution | total file: {len(cache_list)}")

                # Trim to max_file
                if len(cache_list) > max_file:
                    cache_list = cache_list[:max_file]

                # Batch write to disk periodically to reduce I/O overhead
                batch_buffer.append(1)
                if len(batch_buffer) >= batch_size or (time.time() - last_write) >= 2.0:
                    safe_atomic_write_json(outfile, cache_list)
                    last_write = time.time()
                    batch_buffer.clear()

            except Exception as e:
                print("Worker failed:", e)

            i_done += 1

    # Final write
    safe_atomic_write_json(outfile, cache_list)
    elapsed = time.time() - start_all
    print(f"Done. Produced {produced} new puzzles. File now has {len(cache_list)} entries. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
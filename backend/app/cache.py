"""Two-level cache (in-memory LRU + on-disk pickle) for fitted models and forecasts.

Fitting the channel models (hyper-parameter search + inner rolling backtest for
the conformal residuals) dominates request latency. Fits are keyed by
(dataset fingerprint, model, training cut-off, settings), so repeated forecasts,
scenario tweaks and optimiser runs against the same history reuse one fit.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def fingerprint_df(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def make_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class ModelCache:
    def __init__(self, maxsize: int = 128, disk_dir: str | Path | None = None):
        self.maxsize = maxsize
        self.disk_dir = Path(disk_dir) if disk_dir else None
        if self.disk_dir:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._mem: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.Lock] = {}
        self.hits = self.disk_hits = self.misses = 0
        self.compute_seconds = 0.0
        self.saved_seconds = 0.0
        self._cost: dict[str, float] = {}

    def _disk_path(self, key: str) -> Path | None:
        return self.disk_dir / f"{key}.pkl" if self.disk_dir else None

    def get_or_compute(self, namespace: str, key_parts: tuple, fn: Callable[[], Any]) -> Any:
        key = f"{namespace}-{make_key(*key_parts)}"
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                self.hits += 1
                self.saved_seconds += self._cost.get(key, 0.0)
                return self._mem[key]
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        # Per-key lock: concurrent requests for the same fit wait instead of fitting twice.
        with key_lock:
            with self._lock:
                if key in self._mem:
                    self.hits += 1
                    return self._mem[key]
            path = self._disk_path(key)
            if path and path.exists():
                try:
                    value = pickle.loads(path.read_bytes())
                    with self._lock:
                        self.disk_hits += 1
                        self._put(key, value)
                    return value
                except Exception:
                    path.unlink(missing_ok=True)
            t0 = time.perf_counter()
            value = fn()
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.misses += 1
                self.compute_seconds += elapsed
                self._cost[key] = elapsed
                self._put(key, value)
            if path:
                try:
                    path.write_bytes(pickle.dumps(value))
                except Exception:
                    pass
            return value

    def _put(self, key: str, value: Any) -> None:
        self._mem[key] = value
        self._mem.move_to_end(key)
        while len(self._mem) > self.maxsize:
            self._mem.popitem(last=False)

    def clear(self, disk: bool = True) -> None:
        with self._lock:
            self._mem.clear()
            self._cost.clear()
            self.hits = self.disk_hits = self.misses = 0
            self.compute_seconds = self.saved_seconds = 0.0
        if disk and self.disk_dir:
            for p in self.disk_dir.glob("*.pkl"):
                p.unlink(missing_ok=True)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.disk_hits + self.misses
            return {
                "entries": len(self._mem),
                "hits": self.hits,
                "disk_hits": self.disk_hits,
                "misses": self.misses,
                "hit_rate": round((self.hits + self.disk_hits) / total, 3) if total else 0.0,
                "compute_seconds": round(self.compute_seconds, 3),
                "seconds_saved": round(self.saved_seconds, 3),
            }

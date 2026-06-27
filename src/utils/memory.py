"""
MemoryMonitor：GPU 内存监控与缓存清理。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

import torch

from ..configs import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryMonitor:
    def __init__(self, cfg: MemoryConfig) -> None:
        self._max_ratio = cfg.max_gpu_memory_ratio
        self._kv_timeout_s = cfg.kv_cache_timeout_minutes * 60
        self._cleanup_interval = cfg.cleanup_interval
        self._last_cleanup = time.time()
        self._request_ts: Dict[str, float] = {}

        logger.info(
            "MemoryMonitor: max_ratio=%.2f, kv_timeout=%ds",
            self._max_ratio,
            self._kv_timeout_s,
        )

    # ------------------------------------------------------------------ #

    def is_memory_available(self) -> bool:
        if not torch.cuda.is_available():
            return True
        try:
            dev = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(dev).total_memory
            used = torch.cuda.memory_allocated(dev)
            ratio = used / total
            if ratio >= self._max_ratio:
                logger.warning("GPU memory %.1f%% >= threshold %.1f%%", ratio * 100, self._max_ratio * 100)
                return False
            return True
        except Exception as exc:
            logger.error("is_memory_available error: %s", exc)
            return True  # fail-open

    def get_stats(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        try:
            dev = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(dev).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(dev) / 1e9
            cached = torch.cuda.memory_reserved(dev) / 1e9
            return {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "ratio": round(allocated / total, 3),
            }
        except Exception as exc:
            logger.error("get_stats error: %s", exc)
            return {}

    def track_request(self, request_id: str) -> None:
        self._request_ts[request_id] = time.time()

    def cleanup(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_cleanup) < self._cleanup_interval:
            return

        expired = [rid for rid, ts in self._request_ts.items() if now - ts > self._kv_timeout_s]
        for rid in expired:
            del self._request_ts[rid]
        if expired:
            logger.info("Cleaned %d expired request records.", len(expired))

        if torch.cuda.is_available():
            before = self.get_stats().get("cached_gb", 0)
            torch.cuda.empty_cache()
            after = self.get_stats().get("cached_gb", 0)
            logger.info("GPU cache: %.2fGB → %.2fGB", before, after)

        self._last_cleanup = now

    async def run_cleanup_loop(self) -> None:
        """作为 asyncio 后台任务运行。"""
        logger.info("Memory cleanup loop started.")
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self.cleanup()
                logger.debug("Memory stats: %s", self.get_stats())
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Cleanup loop error: %s", exc, exc_info=True)
        logger.info("Memory cleanup loop stopped.")

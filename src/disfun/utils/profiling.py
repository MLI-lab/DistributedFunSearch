"""Profiling decorators for timing and memory measurement.

Time decorators have negligible overhead (microseconds) and can be left enabled.
Memory decorators are expensive as they spawn a subprocess to sample memory usage. Enable only for debugging.

Usage:
    @async_time_execution("Evaluator")
    async def process_message(...):
        ...

    # Memory tracking (expensive, use for debugging only)
    @sync_track_memory("Sampler")
    def generate_sample(...):
        ...
"""
import time
import functools
import logging
from memory_profiler import memory_usage

time_memory_logger = logging.getLogger('time_memory_logger')


def async_time_execution(component: str):
    """Decorator to log execution time of async functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            time_memory_logger.info(f"[{component}] {func.__name__}: {elapsed:.4f}s")
            return result
        return wrapper
    return decorator


def sync_time_execution(component: str):
    """Decorator to log execution time of sync functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            time_memory_logger.info(f"[{component}] {func.__name__}: {elapsed:.4f}s")
            return result
        return wrapper
    return decorator


def async_track_memory(component: str):
    """Decorator to log memory usage of async functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mem_before = memory_usage(-1, interval=0.1, timeout=1)
            result = await func(*args, **kwargs)
            mem_after = memory_usage(-1, interval=0.1, timeout=1)
            mem_used = max(mem_after) - min(mem_before)
            time_memory_logger.info(f"[{component}] {func.__name__}: {mem_used:.2f} MiB")
            return result
        return wrapper
    return decorator


def sync_track_memory(component: str):
    """Decorator to log memory usage of sync functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = memory_usage(-1, interval=0.1, timeout=1)
            result = func(*args, **kwargs)
            mem_after = memory_usage(-1, interval=0.1, timeout=1)
            mem_used = max(mem_after) - min(mem_before)
            time_memory_logger.info(f"[{component}] {func.__name__}: {mem_used:.2f} MiB")
            return result
        return wrapper
    return decorator

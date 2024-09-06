import time
import functools
import logging
from memory_profiler import memory_usage

logger = logging.getLogger('main_logger')

def async_time_execution(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Execution time of {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def sync_time_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Execution time of {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper



def async_track_memory(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
        result = await func(*args, **kwargs)
        mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
        logger.info(f"Memory used by {func.__name__}: {max(mem_usage_after) - min(mem_usage_before)} MiB")
        return result
    return wrapper


def sync_track_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
        result = func(*args, **kwargs)
        mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
        logger.info(f"Memory used by {func.__name__}: {max(mem_usage_after) - min(mem_usage_before)} MiB")
        return result
    return wrapper
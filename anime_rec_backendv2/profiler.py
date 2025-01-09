import time
from functools import wraps
from memory_profiler import memory_usage

def timeit_decorator(func):
    """
    Decorator to measure the execution time of functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {(end_time - start_time):.4f} seconds.")
        return result
    return wrapper

def memory_profiler_decorator(func):
    """
    Decorator to measure the memory usage of functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=None)
        max_mem = max(mem_usage) - min(mem_usage)
        print(f"Function '{func.__name__}' consumed {max_mem:.4f} MiB of memory.")
        return func(*args, **kwargs)
    return wrapper

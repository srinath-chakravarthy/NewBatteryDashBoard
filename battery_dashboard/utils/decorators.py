# battery_dashboard/utils/decorators.py
import functools
import time
import asyncio
from typing import Callable, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to log function execution time"""

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                "Function executed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Function failed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s",
                error=str(e)
            )
            raise

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                "Async function executed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Async function failed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s",
                error=str(e)
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def retry_decorator(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry failed function calls"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Function failed, retrying",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=current_delay,
                            error=str(e)
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "Function failed after all retries",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e)
                        )

            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Async function failed, retrying",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=current_delay,
                            error=str(e)
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "Async function failed after all retries",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e)
                        )

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def cache_result(ttl: Optional[float] = None):
    """Simple in-memory cache decorator"""

    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))

            # Check if cached result exists and is not expired
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug("Cache hit", function=func.__name__, key=key[:50])
                    return result

            # Compute result and cache it
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            logger.debug("Cache miss", function=func.__name__, key=key[:50])
            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "ttl": ttl}

        return wrapper

    return decorator

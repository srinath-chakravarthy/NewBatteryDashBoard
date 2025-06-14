# battery_dashboard/data/cache.py
"""
Caching utilities for battery analytics dashboard.
Provides in-memory caching with TTL, LRU eviction, and thread safety.
"""
import polars as pl
from typing import Dict, Any, Optional, Union, List, Tuple
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import hashlib
import json
import weakref
import gc
from collections import OrderedDict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import asyncio

logger = structlog.get_logger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    LRU_TTL = "lru_ttl"  # Combined LRU and TTL


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None

    def __post_init__(self):
        if self.size_bytes is None:
            self.size_bytes = self._estimate_size()

    def _estimate_size(self) -> int:
        """Estimate memory size of the cached data"""
        try:
            if isinstance(self.data, pl.DataFrame):
                # Estimate DataFrame size
                return self.data.estimated_size()
            elif hasattr(self.data, '__sizeof__'):
                return self.data.__sizeof__()
            else:
                # Rough estimate for other objects
                return len(str(self.data)) * 4  # Assume 4 bytes per character
        except Exception:
            return 1024  # Default fallback

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False

        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class DataCache:
    """
    Thread-safe in-memory cache with TTL and LRU eviction.
    Optimized for Polars DataFrames and battery analytics data.
    """

    def __init__(
            self,
            max_size: int = 50,
            default_ttl: int = 3600,
            strategy: CacheStrategy = CacheStrategy.LRU_TTL,
            max_memory_mb: Optional[int] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'puts': 0,
            'size': 0,
            'memory_bytes': 0
        }

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_cleanup()

        logger.info(
            "DataCache initialized",
            max_size=max_size,
            default_ttl=default_ttl,
            strategy=strategy.value,
            max_memory_mb=max_memory_mb
        )

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a unique cache key from data"""
        try:
            if isinstance(data, (dict, list, tuple)):
                serialized = json.dumps(data, sort_keys=True, default=str)
            else:
                serialized = str(data)

            hash_obj = hashlib.md5(serialized.encode())
            return f"{prefix}:{hash_obj.hexdigest()}"
        except Exception:
            # Fallback to simple string representation
            return f"{prefix}:{hash(str(data))}"

    def get(
            self,
            key: str,
            ttl: Optional[int] = None
    ) -> Optional[Any]:
        """
        Retrieve item from cache.

        Args:
            key: Cache key
            ttl: Override TTL check (None uses entry's TTL)

        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                logger.debug("Cache miss", key=key)
                return None

            entry = self._cache[key]

            # Check TTL
            effective_ttl = ttl if ttl is not None else self.default_ttl
            if effective_ttl > 0:
                max_age = effective_ttl
                if entry.age_seconds > max_age:
                    logger.debug("Cache entry expired", key=key, age=entry.age_seconds)
                    del self._cache[key]
                    self._stats['misses'] += 1
                    self._update_memory_stats()
                    return None

            # Move to end for LRU
            self._cache.move_to_end(key)
            entry.touch()

            self._stats['hits'] += 1
            logger.debug(
                "Cache hit",
                key=key,
                age=entry.age_seconds,
                access_count=entry.access_count
            )

            return entry.data

    def put(
            self,
            key: str,
            data: Any,
            ttl: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store item in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        if data is None:
            return False

        with self._lock:
            now = datetime.now()
            entry = CacheEntry(
                data=data,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl or self.default_ttl,
                metadata=metadata or {}
            )

            # Check memory limits
            if self.max_memory_bytes and entry.size_bytes:
                if self._get_total_memory() + entry.size_bytes > self.max_memory_bytes:
                    logger.warning(
                        "Cache memory limit would be exceeded",
                        current_memory=self._get_total_memory(),
                        new_entry_size=entry.size_bytes,
                        limit=self.max_memory_bytes
                    )
                    self._evict_by_memory()

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Add new entry
            self._cache[key] = entry
            self._cache.move_to_end(key)

            self._stats['puts'] += 1
            self._update_memory_stats()

            # Evict if necessary
            self._evict_if_needed()

            logger.debug(
                "Cache put",
                key=key,
                size_bytes=entry.size_bytes,
                ttl=entry.ttl_seconds,
                cache_size=len(self._cache)
            )

            return True

    def delete(self, key: str) -> bool:
        """Delete specific cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._update_memory_stats()
                logger.debug("Cache delete", key=key)
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            size_before = len(self._cache)
            self._cache.clear()
            self._stats['size'] = 0
            self._stats['memory_bytes'] = 0
            logger.info("Cache cleared", entries_removed=size_before)

    def _evict_if_needed(self):
        """Evict entries based on strategy"""
        if len(self._cache) <= self.max_size:
            return

        entries_to_remove = len(self._cache) - self.max_size

        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(entries_to_remove)
        elif self.strategy == CacheStrategy.TTL:
            self._evict_expired()
        elif self.strategy == CacheStrategy.LRU_TTL:
            # First remove expired, then LRU if needed
            self._evict_expired()
            if len(self._cache) > self.max_size:
                self._evict_lru(len(self._cache) - self.max_size)

    def _evict_lru(self, count: int):
        """Evict least recently used entries"""
        keys_to_remove = list(self._cache.keys())[:count]
        for key in keys_to_remove:
            del self._cache[key]
            self._stats['evictions'] += 1

        logger.debug("LRU eviction", removed_count=count)

    def _evict_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        keys_to_remove = []

        for key, entry in self._cache.items():
            if entry.is_expired:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]
            self._stats['evictions'] += 1

        if keys_to_remove:
            logger.debug("TTL eviction", removed_count=len(keys_to_remove))

    def _evict_by_memory(self):
        """Evict entries to free memory"""
        if not self.max_memory_bytes:
            return

        target_memory = self.max_memory_bytes * 0.8  # Evict to 80% of limit
        current_memory = self._get_total_memory()

        # Sort by last accessed time (oldest first)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )

        for key, entry in sorted_items:
            if current_memory <= target_memory:
                break

            del self._cache[key]
            current_memory -= entry.size_bytes or 0
            self._stats['evictions'] += 1

        logger.info(
            "Memory-based eviction completed",
            memory_before=self._get_total_memory(),
            memory_after=current_memory,
            target=target_memory
        )

    def _get_total_memory(self) -> int:
        """Calculate total memory usage"""
        return sum(
            entry.size_bytes or 0
            for entry in self._cache.values()
        )

    def _update_memory_stats(self):
        """Update memory statistics"""
        self._stats['size'] = len(self._cache)
        self._stats['memory_bytes'] = self._get_total_memory()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = (
                self._stats['hits'] / (self._stats['hits'] + self._stats['misses'])
                if (self._stats['hits'] + self._stats['misses']) > 0
                else 0
            )

            return {
                **self._stats,
                'hit_rate': hit_rate,
                'memory_mb': self._stats['memory_bytes'] / (1024 * 1024),
                'average_entry_size': (
                    self._stats['memory_bytes'] / self._stats['size']
                    if self._stats['size'] > 0
                    else 0
                )
            }

    def get_cache_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about cached entries"""
        with self._lock:
            info = []
            for key, entry in self._cache.items():
                info.append({
                    'key': key,
                    'created_at': entry.created_at.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat(),
                    'age_seconds': entry.age_seconds,
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes,
                    'ttl_seconds': entry.ttl_seconds,
                    'is_expired': entry.is_expired,
                    'metadata': entry.metadata
                })
            return info

    def _start_background_cleanup(self):
        """Start background cleanup task"""
        try:
            # Only start if we have a running event loop
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._background_cleanup())
        except RuntimeError:
            # No event loop or not running - skip background cleanup
            self._cleanup_task = None
            logger.debug("No event loop available, skipping background cache cleanup")
        except Exception as e:
            logger.warning(f"Could not start background cleanup: {e}")
            self._cleanup_task = None

    async def _background_cleanup(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                with self._lock:
                    self._evict_expired()

                # Force garbage collection periodically
                gc.collect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Background cleanup error", error=str(e))


class CacheManager:
    """
    High-level cache manager for the application.
    Manages multiple named caches with different configurations.
    """

    def __init__(self):
        self._caches: Dict[str, DataCache] = {}
        self._default_configs = {
            'cell_data': {
                'max_size': 20,
                'default_ttl': 1800,  # 30 minutes
                'max_memory_mb': 100
            },
            'cycle_data': {
                'max_size': 100,
                'default_ttl': 900,  # 15 minutes
                'max_memory_mb': 500
            },
            'query_results': {
                'max_size': 50,
                'default_ttl': 600,  # 10 minutes
                'max_memory_mb': 200
            },
            'processed_data': {
                'max_size': 30,
                'default_ttl': 1200,  # 20 minutes
                'max_memory_mb': 300
            }
        }

        logger.info("CacheManager initialized")

    def get_cache(self, name: str) -> DataCache:
        """Get or create a named cache"""
        if name not in self._caches:
            config = self._default_configs.get(name, {})
            self._caches[name] = DataCache(**config)
            logger.info("Created cache", name=name, config=config)

        return self._caches[name]

    def clear_all_caches(self):
        """Clear all managed caches"""
        for name, cache in self._caches.items():
            cache.clear()
        logger.info("All caches cleared")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        stats = {}
        total_memory = 0
        total_entries = 0

        for name, cache in self._caches.items():
            cache_stats = cache.get_stats()
            stats[name] = cache_stats
            total_memory += cache_stats['memory_bytes']
            total_entries += cache_stats['size']

        stats['global'] = {
            'total_caches': len(self._caches),
            'total_entries': total_entries,
            'total_memory_mb': total_memory / (1024 * 1024)
        }

        return stats


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions
def get_cell_cache() -> DataCache:
    """Get the cell data cache"""
    return cache_manager.get_cache('cell_data')


def get_cycle_cache() -> DataCache:
    """Get the cycle data cache"""
    return cache_manager.get_cache('cycle_data')


def get_query_cache() -> DataCache:
    """Get the query results cache"""
    return cache_manager.get_cache('query_results')


def get_processed_cache() -> DataCache:
    """Get the processed data cache"""
    return cache_manager.get_cache('processed_data')


# Decorators
def cache_result(cache_name: str, ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator to cache function results.

    Args:
        cache_name: Name of cache to use
        ttl: Time-to-live for cached result
        key_prefix: Prefix for cache key
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name)

            # Generate cache key
            key_data = {'args': args, 'kwargs': kwargs}
            cache_key = cache._generate_key(key_prefix or func.__name__, key_data)

            # Try to get from cache
            result = cache.get(cache_key, ttl=ttl)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


# Cache warming functions
async def warm_cache_for_cells(cell_ids: List[int]):
    """Pre-load cache with data for specified cells"""
    from .loaders import get_cycle_data

    logger.info("Warming cache for cells", cell_count=len(cell_ids))

    # Load data in background
    cycle_cache = get_cycle_cache()

    try:
        # This will populate the cache
        await get_cycle_data(cell_ids)
        logger.info("Cache warming completed", cell_count=len(cell_ids))
    except Exception as e:
        logger.error("Cache warming failed", error=str(e))
# battery_dashboard/core/data_manager.py
import asyncio
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl
import param
import threading
from concurrent.futures import ThreadPoolExecutor

from ..data.loaders import get_redash_query_results, CELL_QUERY_ID, CYCLE_QUERY_ID
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for data storage"""
    data: pl.DataFrame
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired"""
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds

    def access(self):
        """Mark this entry as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class DataCache:
    """Intelligent caching system for data"""

    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[pl.DataFrame]:
        """Get data from cache"""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            ttl = ttl or self.default_ttl

            if entry.is_expired(ttl):
                del self._cache[key]
                return None

            entry.access()
            return entry.data.clone()

    def put(self, key: str, data: pl.DataFrame, metadata: Optional[Dict] = None):
        """Store data in cache"""
        with self._lock:
            # Evict expired entries
            self._evict_expired()

            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                data=data.clone(),
                timestamp=datetime.now(),
                metadata=metadata or {}
            )

    def _evict_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired(self.default_ttl)
        ]
        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self):
        """Remove least recently used entry"""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            total_access_count = sum(entry.access_count for entry in self._cache.values())

            return {
                'total_entries': total_entries,
                'max_size': self.max_size,
                'total_accesses': total_access_count,
                'cache_keys': list(self._cache.keys())
            }


class LoadingState(param.Parameterized):
    """Reactive loading state tracker"""
    is_loading = param.Boolean(default=False)
    progress = param.Number(default=0.0, bounds=(0.0, 1.0))
    message = param.String(default="")
    error = param.String(default="")

    def start_loading(self, message: str = "Loading..."):
        """Start loading state"""
        self.is_loading = True
        self.progress = 0.0
        self.message = message
        self.error = ""

    def update_progress(self, progress: float, message: str = None):
        """Update loading progress"""
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message

    def finish_loading(self, message: str = "Complete"):
        """Finish loading state"""
        self.is_loading = False
        self.progress = 1.0
        self.message = message

    def set_error(self, error: str):
        """Set error state"""
        self.is_loading = False
        self.error = error
        self.message = "Error occurred"


class CellDataManager(param.Parameterized):
    """Manager for cell metadata and basic cell information"""

    data = param.Parameter(default=None)
    loading_state = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self.cache = DataCache(max_size=10, default_ttl=600)  # 10 minute TTL for cell data
        self.loading_state = LoadingState()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._data = None

    async def load_initial_data(self, filters: Optional[Dict] = None, force_refresh: bool = False):
        """Load initial cell data with optional filters"""
        cache_key = self.cache._generate_key("cell_data", filters)
        logger.info(f"Cache stats before lookup: {self.cache.get_stats()}")
        logger.info(f"Looking for cache key: {cache_key}")
        # Check cache first
        if not force_refresh:
            cached_data = self.cache.get(cache_key, ttl=600)
            if cached_data is not None:
                self._data = cached_data
                self.data = cached_data
                logger.info(f"Loaded cell data from cache: {len(cached_data)} rows")
                return cached_data

        # Load from source
        self.loading_state.start_loading("Loading cell data...")

        try:
            # Run blocking operation in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor,
                self._fetch_cell_data,
                filters
            )

            if data is not None and not data.is_empty():
                # Apply any initial filtering
                if 'total_cycles' in data.columns:
                    data = data.filter(pl.col('total_cycles') > 2)

                # Cache the result
                self.cache.put(cache_key, data, metadata={'filters': filters})

                self._data = data
                self.data = data
                self.loading_state.finish_loading(f"Loaded {len(data)} cells")
                logger.info(f"Loaded cell data: {len(data)} rows")
                return data
            else:
                self.loading_state.set_error("No cell data returned")
                return pl.DataFrame()

        except Exception as e:
            logger.error(f"Error loading cell data: {e}")
            self.loading_state.set_error(str(e))
            return pl.DataFrame()

    def _fetch_cell_data(self, filters: Optional[Dict] = None) -> pl.DataFrame:
        """Fetch cell data (blocking operation)"""
        try:
            result =  get_redash_query_results(CELL_QUERY_ID)
            logger.info(f"Raw query result: {len(result)} rows, columns: {result.columns if not result.is_empty() else 'None'}")
            return result
        except Exception as e:
            logger.error(f"Error fetching cell data: {e}")
            raise

    def get_filtered_data(self, filters: Dict[str, Any]) -> pl.DataFrame:
        """Get filtered cell data"""
        if self._data is None:
            return pl.DataFrame()

        filtered_data = self._data.clone()

        for column, value in filters.items():
            if value and column in filtered_data.columns:
                filtered_data = filtered_data.filter(pl.col(column) == value)

        return filtered_data

    def search_cells(self, query: str) -> pl.DataFrame:
        """Search cells based on query string"""
        if self._data is None or not query:
            return self._data or pl.DataFrame()

        # Implement search logic similar to your current implementation
        # This is a simplified version
        search_columns = ['cell_name', 'cell_type', 'description', 'notes']

        search_filters = []
        for col in search_columns:
            if col in self._data.columns:
                search_filters.append(
                    pl.col(col).cast(pl.Utf8).str.contains(query, literal=True)
                )

        if search_filters:
            combined_filter = search_filters[0]
            for filter_expr in search_filters[1:]:
                combined_filter = combined_filter | filter_expr

            return self._data.filter(combined_filter)

        return self._data

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


class CycleDataManager(param.Parameterized):
    """Manager for cycle-level battery data"""

    loading_state = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self.cache = DataCache(max_size=50, default_ttl=900)  # 15 minute TTL for cycle data
        self.loading_state = LoadingState()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._active_requests: Dict[str, asyncio.Task] = {}

    async def get_cycle_data(
            self,
            cell_ids: List[int],
            cell_metadata: Optional[pl.DataFrame] = None,
            force_refresh: bool = False,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pl.DataFrame:
        """Get cycle data for specified cell IDs"""
        if not cell_ids:
            return pl.DataFrame()

        cache_key = self.cache._generate_key("cycle_data", tuple(sorted(cell_ids)))

        # Check if request is already in progress
        if cache_key in self._active_requests:
            logger.info(f"Request already in progress for {len(cell_ids)} cells")
            return await self._active_requests[cache_key]

        # Check cache first
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded cycle data from cache: {len(cached_data)} rows for {len(cell_ids)} cells")
                return await self._add_derived_columns(cached_data, cell_metadata)

        # Create and store the request task
        task = asyncio.create_task(
            self._fetch_cycle_data_async(cell_ids, cell_metadata, progress_callback)
        )
        self._active_requests[cache_key] = task

        try:
            result = await task
            # Cache the result
            if result is not None and not result.is_empty():
                # Store base data without derived columns for caching
                base_data = self._get_base_cycle_data(result)
                self.cache.put(cache_key, base_data, metadata={'cell_ids': cell_ids})

            return result
        finally:
            # Remove from active requests
            if cache_key in self._active_requests:
                del self._active_requests[cache_key]

    async def _fetch_cycle_data_async(
            self,
            cell_ids: List[int],
            cell_metadata: Optional[pl.DataFrame] = None,
            progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pl.DataFrame:
        """Fetch cycle data asynchronously"""
        self.loading_state.start_loading(f"Loading cycle data for {len(cell_ids)} cells...")

        try:
            # Batch process cells to avoid overwhelming the API
            batch_size = 10
            all_results = []

            for i in range(0, len(cell_ids), batch_size):
                batch_cell_ids = cell_ids[i:i + batch_size]
                progress = i / len(cell_ids)

                if progress_callback:
                    progress_callback(progress, f"Processing batch {i // batch_size + 1}")

                self.loading_state.update_progress(
                    progress,
                    f"Loading batch {i // batch_size + 1} of {(len(cell_ids) + batch_size - 1) // batch_size}"
                )

                # Process batch in parallel
                loop = asyncio.get_event_loop()
                batch_tasks = [
                    loop.run_in_executor(self.executor, self._fetch_single_cell_data, cell_id)
                    for cell_id in batch_cell_ids
                ]

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Filter out exceptions and empty results
                valid_results = [
                    result for result in batch_results
                    if not isinstance(result, Exception) and result is not None and not result.is_empty()
                ]

                all_results.extend(valid_results)

            if all_results:
                # Combine all results
                combined_data = pl.concat(all_results)

                # Add derived columns
                result = await self._add_derived_columns(combined_data, cell_metadata)

                self.loading_state.finish_loading(f"Loaded {len(result)} cycles for {len(cell_ids)} cells")
                return result
            else:
                self.loading_state.set_error("No cycle data found")
                return pl.DataFrame()

        except Exception as e:
            logger.error(f"Error loading cycle data: {e}")
            self.loading_state.set_error(str(e))
            return pl.DataFrame()

    def _fetch_single_cell_data(self, cell_id: int) -> pl.DataFrame:
        """Fetch data for a single cell (blocking operation)"""
        try:
            params = {"cell_ids": str(cell_id)}
            data = get_redash_query_results(CYCLE_QUERY_ID, params)

            if data is not None and not data.is_empty():
                # Add basic normalizations here if needed
                return self._add_basic_normalizations(data)

            return pl.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for cell {cell_id}: {e}")
            return pl.DataFrame()

    def _add_basic_normalizations(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add basic cycle normalizations"""
        if data.is_empty():
            return data

        # List of columns that should be normalized
        normalize_columns = [
            col for col in data.columns
            if any(term in col.lower() for term in ['_capacity', '_energy'])
        ]

        if not normalize_columns:
            return data

        # Create expressions for normalization
        norm_expressions = []

        for col in normalize_columns:
            # Regular cycle normalization (using cycle 1 as reference)
            try:
                # Get first cycle value for this cell
                first_cycle_data = data.filter(pl.col('regular_cycle_number') == 1)
                if not first_cycle_data.is_empty():
                    first_cycle_val = first_cycle_data[col].first()
                    if first_cycle_val and first_cycle_val > 0:
                        norm_expressions.append(
                            (pl.col(col) / first_cycle_val).alias(f'{col}_norm_reg')
                        )

                # 95th percentile normalization
                p95_val = data[col].quantile(0.95)
                if p95_val and p95_val > 0:
                    norm_expressions.append(
                        (pl.col(col) / p95_val).alias(f'{col}_norm_p95')
                    )
            except Exception as e:
                logger.warning(f"Error normalizing column {col}: {e}")
                continue

        # Apply normalizations
        if norm_expressions:
            data = data.with_columns(norm_expressions)

        return data

    async def _add_derived_columns(
            self,
            cycle_data: pl.DataFrame,
            cell_metadata: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Add derived columns like specific capacity and energy"""
        if cycle_data.is_empty():
            return cycle_data

        result = cycle_data.clone()

        # Merge with cell metadata if available
        if cell_metadata is not None and not cell_metadata.is_empty():
            # Find columns to merge (avoid duplicates)
            existing_cols = set(result.columns)
            metadata_cols = [col for col in cell_metadata.columns if col != 'cell_id']
            unique_metadata_cols = [col for col in metadata_cols if col not in existing_cols]

            if unique_metadata_cols:
                cols_to_select = ['cell_id'] + unique_metadata_cols
                result = result.join(
                    cell_metadata.select(cols_to_select),
                    on='cell_id',
                    how='left'
                )

        # Add specific capacity and energy if active mass is available
        if 'total_active_mass_g' in result.columns:
            result = self._add_specific_metrics(result)

        return result

    def _add_specific_metrics(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add specific capacity and energy metrics"""
        capacity_cols = [
            col for col in data.columns
            if 'capacity' in col.lower() and not '_specific_mAh_g' in col
        ]
        energy_cols = [
            col for col in data.columns
            if 'energy' in col.lower() and not '_specific_Wh_g' in col
        ]

        specific_exprs = []

        # Add expressions for capacity columns (convert to mAh/g)
        for col in capacity_cols:
            specific_exprs.append(
                (
                    pl.when(pl.col('total_active_mass_g') > 0)
                    .then(1000 * pl.col(col) / pl.col('total_active_mass_g'))
                    .otherwise(None)
                ).alias(f"{col}_specific_mAh_g")
            )

        # Add expressions for energy columns (convert to Wh/g)
        for col in energy_cols:
            specific_exprs.append(
                (
                    pl.when(pl.col('total_active_mass_g') > 0)
                    .then(pl.col(col) / pl.col('total_active_mass_g'))
                    .otherwise(None)
                ).alias(f"{col}_specific_Wh_g")
            )

        # Apply the expressions
        if specific_exprs:
            data = data.with_columns(specific_exprs)

        return data

    def _get_base_cycle_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Get base cycle data without derived columns for caching"""
        # Remove derived columns that are added dynamically
        base_columns = [
            col for col in data.columns
            if not col.endswith('_specific_mAh_g') and not col.endswith('_specific_Wh_g')
        ]
        return data.select(base_columns)

    def invalidate_cache(self, cell_ids: Optional[List[int]] = None):
        """Invalidate cache entries"""
        if cell_ids is None:
            self.cache.clear()
        else:
            # Invalidate specific entries (simplified - could be more targeted)
            cache_key = self.cache._generate_key("cycle_data", tuple(sorted(cell_ids)))
            with self.cache._lock:
                if cache_key in self.cache._cache:
                    del self.cache._cache[cache_key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


# Global data manager instances
cell_data_manager = CellDataManager()
cycle_data_manager = CycleDataManager()
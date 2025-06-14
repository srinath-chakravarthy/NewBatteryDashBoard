# battery_dashboard/data/loaders.py
"""
Data loading utilities for battery analytics dashboard.
Handles data retrieval from various sources including Redash API.
"""
import polars as pl
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import structlog
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..config import REDASH_URL, REDASH_API_KEY, CELL_QUERY_ID, CYCLE_QUERY_ID
from ..core.exceptions import DataLoadingError, DataValidationError
from ..utils.decorators import timing_decorator, retry_decorator
from ..utils.helpers import validate_dataframe_schema

logger = structlog.get_logger(__name__)

# Cache for query results with timestamps
_query_cache: Dict[str, Dict[str, Any]] = {}


class RedashClient:
    """Client for interacting with Redash API"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json"
        })

    @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    @timing_decorator
    def execute_query(
            self,
            query_id: Union[str, int],
            parameters: Optional[Dict[str, Any]] = None,
            timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Execute a Redash query and return the results.

        Args:
            query_id: The Redash query ID
            parameters: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Query result data

        Raises:
            DataLoadingError: If query execution fails
        """
        try:
            api_url = f"{self.base_url}/api/queries/{query_id}/results"
            payload = {"parameters": parameters or {}}

            logger.info(
                "Executing Redash query",
                query_id=query_id,
                parameters=parameters,
                url=api_url
            )

            response = self.session.post(
                api_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()

            if "query_result" not in result:
                raise DataLoadingError(f"Invalid response format from query {query_id}")

            if "data" not in result["query_result"]:
                raise DataLoadingError(f"No data in query result for query {query_id}")

            rows = result["query_result"]["data"].get("rows", [])
            columns = result["query_result"]["data"].get("columns", [])

            logger.info(
                "Query executed successfully",
                query_id=query_id,
                rows_returned=len(rows),
                columns_returned=len(columns)
            )

            return result["query_result"]["data"]

        except requests.exceptions.RequestException as e:
            logger.error(
                "HTTP error executing query",
                query_id=query_id,
                error=str(e),
                status_code=getattr(e.response, 'status_code', None)
            )
            raise DataLoadingError(f"Failed to execute query {query_id}: {str(e)}") from e
        except Exception as e:
            logger.error(
                "Unexpected error executing query",
                query_id=query_id,
                error=str(e)
            )
            raise DataLoadingError(f"Unexpected error in query {query_id}: {str(e)}") from e


def _generate_cache_key(query_id: Union[str, int], parameters: Optional[Dict] = None) -> str:
    """Generate a cache key for the given query and parameters."""
    params_str = json.dumps(parameters or {}, sort_keys=True)
    return f"{query_id}_{hash(params_str)}"


def _is_cache_valid(cache_entry: Dict[str, Any], ttl_seconds: int) -> bool:
    """Check if a cache entry is still valid."""
    if "timestamp" not in cache_entry:
        return False

    cache_time = datetime.fromisoformat(cache_entry["timestamp"])
    return (datetime.now() - cache_time).total_seconds() < ttl_seconds


@timing_decorator
def get_redash_query_results(
        query_id: Union[str, int],
        parameters: Optional[Dict[str, Any]] = None,
        cache_ttl: int = 300,
        use_cache: bool = True
) -> pl.DataFrame:
    """
    Fetch results from a Redash query with caching and error handling.

    Args:
        query_id: Redash query ID
        parameters: Query parameters
        cache_ttl: Cache time-to-live in seconds
        use_cache: Whether to use caching

    Returns:
        Polars DataFrame with query results

    Raises:
        DataLoadingError: If query fails or returns invalid data
    """
    cache_key = _generate_cache_key(query_id, parameters)

    # Check cache first
    if use_cache and cache_key in _query_cache:
        cache_entry = _query_cache[cache_key]
        if _is_cache_valid(cache_entry, cache_ttl):
            logger.info("Loading data from cache", query_id=query_id, cache_key=cache_key)
            return cache_entry["data"]

    # Initialize Redash client
    client = RedashClient(REDASH_URL, REDASH_API_KEY)

    try:
        # Execute query
        query_data = client.execute_query(query_id, parameters)

        # Convert to Polars DataFrame
        if not query_data.get("rows"):
            logger.warning("Query returned no rows", query_id=query_id)
            return pl.DataFrame()

        df = pl.DataFrame(query_data["rows"])

        # Validate basic structure
        if df.is_empty():
            logger.warning("Created empty DataFrame", query_id=query_id)
            return df

        # Cache the result
        if use_cache:
            _query_cache[cache_key] = {
                "data": df,
                "timestamp": datetime.now().isoformat(),
                "query_id": query_id,
                "parameters": parameters
            }
            logger.info("Cached query result", query_id=query_id, cache_key=cache_key)

        return df

    except Exception as e:
        logger.error(
            "Failed to load data from Redash",
            query_id=query_id,
            parameters=parameters,
            error=str(e)
        )
        raise DataLoadingError(f"Failed to load data from query {query_id}") from e


@timing_decorator
def load_initial_data(filters: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
    """
    Load initial cell data with optional filtering.

    Args:
        filters: Optional filters to apply

    Returns:
        Polars DataFrame with cell data

    Raises:
        DataLoadingError: If data loading fails
        DataValidationError: If data validation fails
    """
    logger.info("Loading initial cell data", filters=filters)

    try:
        # Load cell data
        df = get_redash_query_results(CELL_QUERY_ID, parameters=filters)

        if df.is_empty():
            logger.warning("No cell data returned")
            return df

        # Validate required columns
        required_columns = ["cell_id", "cell_name"]
        validation_result = validate_dataframe_schema(df, required_columns)

        if not validation_result["valid"]:
            raise DataValidationError(f"Cell data validation failed: {validation_result['errors']}")

        # Apply basic filtering
        if "regular_cycles" in df.columns:
            initial_count = len(df)
            df = df.filter(pl.col("regular_cycles") > 20)
            filtered_count = len(df)

            logger.info(
                "Applied cycle count filter",
                initial_rows=initial_count,
                filtered_rows=filtered_count,
                removed_rows=initial_count - filtered_count
            )

        logger.info("Cell data loaded successfully", rows=len(df), columns=len(df.columns))
        return df

    except Exception as e:
        logger.error("Failed to load initial data", error=str(e))
        raise


@timing_decorator
def get_cycle_data(
        cell_ids: List[int],
        cell_metadata: Optional[pl.DataFrame] = None,
        batch_size: int = 10,
        max_workers: int = 4
) -> pl.DataFrame:
    """
    Get cycle data for specified cell IDs with parallel processing.

    Args:
        cell_ids: List of cell IDs to fetch data for
        cell_metadata: Optional cell metadata for enrichment
        batch_size: Number of cells to process in each batch
        max_workers: Maximum number of worker threads

    Returns:
        Polars DataFrame with cycle data

    Raises:
        DataLoadingError: If data loading fails
    """
    if not cell_ids:
        logger.warning("No cell IDs provided")
        return pl.DataFrame()

    logger.info(
        "Loading cycle data",
        cell_count=len(cell_ids),
        batch_size=batch_size,
        max_workers=max_workers
    )

    def fetch_single_cell_data(cell_id: int) -> Optional[pl.DataFrame]:
        """Fetch data for a single cell."""
        try:
            params = {"cell_ids": str(cell_id)}
            cell_data = get_redash_query_results(CYCLE_QUERY_ID, params)

            if cell_data.is_empty():
                logger.warning("No cycle data for cell", cell_id=cell_id)
                return None

            # Add basic normalizations
            cell_data = _add_basic_normalizations(cell_data)
            return cell_data

        except Exception as e:
            logger.error("Failed to fetch data for cell", cell_id=cell_id, error=str(e))
            return None

    # Process cells in parallel
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_cell = {
            executor.submit(fetch_single_cell_data, cell_id): cell_id
            for cell_id in cell_ids
        }

        # Collect results
        for future in future_to_cell:
            cell_id = future_to_cell[future]
            try:
                result = future.result(timeout=60)  # 60 second timeout per cell
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.error("Future failed for cell", cell_id=cell_id, error=str(e))

    if not all_results:
        logger.warning("No cycle data retrieved for any cells")
        return pl.DataFrame()

    # Combine all results
    try:
        combined_data = pl.concat(all_results)
        logger.info("Cycle data combined successfully", total_rows=len(combined_data))

        # Add derived columns if metadata is available
        if cell_metadata is not None:
            combined_data = _merge_cell_metadata(combined_data, cell_metadata)

        return combined_data

    except Exception as e:
        logger.error("Failed to combine cycle data", error=str(e))
        raise DataLoadingError("Failed to combine cycle data") from e


def _add_basic_normalizations(data: pl.DataFrame) -> pl.DataFrame:
    """Add basic cycle normalizations to the data."""
    if data.is_empty():
        return data

    # Find columns that should be normalized
    normalize_columns = [
        col for col in data.columns
        if any(term in col.lower() for term in ['_capacity', '_energy'])
    ]

    if not normalize_columns:
        return data

    norm_expressions = []

    for col in normalize_columns:
        try:
            # Regular cycle normalization (using cycle 1 as reference)
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
            logger.warning("Error normalizing column", column=col, error=str(e))
            continue

    # Apply normalizations
    if norm_expressions:
        data = data.with_columns(norm_expressions)
        logger.info("Added normalizations", columns=len(norm_expressions))

    return data


def _merge_cell_metadata(cycle_data: pl.DataFrame, cell_metadata: pl.DataFrame) -> pl.DataFrame:
    """Merge cycle data with cell metadata."""
    try:
        # Find columns to merge (avoid duplicates)
        existing_cols = set(cycle_data.columns)
        metadata_cols = [col for col in cell_metadata.columns if col != 'cell_id']
        unique_metadata_cols = [col for col in metadata_cols if col not in existing_cols]

        if not unique_metadata_cols:
            logger.info("No unique metadata columns to merge")
            return cycle_data

        # Perform the join
        cols_to_select = ['cell_id'] + unique_metadata_cols
        merged_data = cycle_data.join(
            cell_metadata.select(cols_to_select),
            on='cell_id',
            how='left'
        )

        logger.info(
            "Merged cell metadata",
            original_columns=len(cycle_data.columns),
            added_columns=len(unique_metadata_cols),
            final_columns=len(merged_data.columns)
        )

        return merged_data

    except Exception as e:
        logger.error("Failed to merge cell metadata", error=str(e))
        return cycle_data


# Cache management functions
def clear_query_cache():
    """Clear the query result cache."""
    global _query_cache
    cache_size = len(_query_cache)
    _query_cache.clear()
    logger.info("Cleared query cache", cached_entries=cache_size)


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the query cache."""
    cache_entries = len(_query_cache)
    total_size = sum(
        len(entry["data"]) for entry in _query_cache.values()
        if "data" in entry and hasattr(entry["data"], "__len__")
    )

    return {
        "cache_entries": cache_entries,
        "total_cached_rows": total_size,
        "cache_keys": list(_query_cache.keys())
    }


# Configuration validation
def validate_data_source_config() -> Dict[str, bool]:
    """Validate that data source configuration is correct."""
    config_status = {
        "redash_url_set": bool(REDASH_URL),
        "redash_api_key_set": bool(REDASH_API_KEY),
        "cell_query_id_set": bool(CELL_QUERY_ID),
        "cycle_query_id_set": bool(CYCLE_QUERY_ID),
    }

    # Test connection if all config is present
    if all(config_status.values()):
        try:
            client = RedashClient(REDASH_URL, REDASH_API_KEY)
            # Try a simple query to test connectivity
            test_result = client.execute_query(CELL_QUERY_ID, {})
            config_status["redash_connectivity"] = True
        except Exception as e:
            logger.error("Redash connectivity test failed", error=str(e))
            config_status["redash_connectivity"] = False
    else:
        config_status["redash_connectivity"] = False

    return config_status
# battery_dashboard/data/__init__.py
"""
Data module for battery analytics dashboard.

This module provides comprehensive data handling capabilities including:
- Data loading from various sources (Redash, databases, files)
- Data processing and transformation pipelines
- Caching with TTL and LRU eviction
- Data validation and quality checks

Main components:
- loaders: Raw data loading utilities
- processors: Data transformation and feature engineering
- cache: Intelligent caching with memory management
- validators: Data quality validation and schema checks
"""

from typing import Dict, List, Optional, Any
import polars as pl
import datetime
import structlog

# Import main classes and functions from each module
from .loaders import (
    RedashClient,
    get_redash_query_results,
    load_initial_data,
    get_cycle_data,
    clear_query_cache,
    get_cache_stats,
    validate_data_source_config
)

from .processors import (
    DataProcessor,
    ProcessingConfig,
    NormalizationMethod,
    process_battery_data,
    create_processing_config,
    add_specific_capacity_energy,
    calculate_coulombic_efficiency,
    add_temperature_features
)

from .cache import (
    DataCache,
    CacheManager,
    CacheStrategy,
    CacheEntry,
    cache_manager,
    get_cell_cache,
    get_cycle_cache,
    get_query_cache,
    get_processed_cache,
    cache_result,
    warm_cache_for_cells
)

from .validators import (
    DataValidator,
    ValidationRule,
    ValidationResult,
    DataQualityReport,
    ValidationType,
    Severity,
    validate_battery_data,
    validate_cell_metadata,
    validate_cycle_data,
    quick_data_check,
    calculate_data_completeness,
    calculate_data_consistency
)

logger = structlog.get_logger(__name__)

# Version and module info
__version__ = "2.0.0"
__author__ = "Battery Analytics Team"


# Main data pipeline functions
def load_and_process_data(
        cell_ids: Optional[List[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
        processing_config: Optional[ProcessingConfig] = None,
        validate: bool = True,
        use_cache: bool = True
) -> Dict[str, Any]:
    """
    Main data pipeline: load, process, and validate battery data.

    Args:
        cell_ids: Specific cell IDs to load (None for all)
        filters: Filters to apply during loading
        processing_config: Data processing configuration
        validate: Whether to run data validation
        use_cache: Whether to use caching

    Returns:
        Dictionary with processed data and metadata
    """
    logger.info(
        "Starting data pipeline",
        cell_ids_count=len(cell_ids) if cell_ids else "all",
        filters=filters,
        validate=validate,
        use_cache=use_cache
    )

    result = {
        'cell_data': None,
        'cycle_data': None,
        'validation_report': None,
        'processing_stats': {},
        'cache_stats': {},
        'errors': []
    }

    try:
        # Step 1: Load cell metadata
        logger.info("Loading cell metadata")
        cell_data = load_initial_data(filters=filters)

        if cell_data.is_empty():
            logger.warning("No cell data loaded")
            result['errors'].append("No cell data available")
            return result

        result['cell_data'] = cell_data

        # Step 2: Load cycle data
        if cell_ids is None:
            # Use all available cell IDs
            cell_ids = cell_data['cell_id'].unique().to_list()

        logger.info("Loading cycle data", cell_count=len(cell_ids))
        cycle_data = get_cycle_data(cell_ids, cell_metadata=cell_data)

        if cycle_data.is_empty():
            logger.warning("No cycle data loaded")
            result['errors'].append("No cycle data available")
            return result

        # Step 3: Process data
        logger.info("Processing cycle data")
        processed_data = process_battery_data(
            cycle_data,
            cell_metadata=cell_data,
            config=processing_config
        )

        result['cycle_data'] = processed_data
        result['processing_stats'] = {
            'original_rows': len(cycle_data),
            'processed_rows': len(processed_data),
            'original_columns': len(cycle_data.columns),
            'processed_columns': len(processed_data.columns),
            'added_columns': len(processed_data.columns) - len(cycle_data.columns)
        }

        # Step 4: Validate data if requested
        if validate:
            logger.info("Validating processed data")
            validation_report = validate_battery_data(processed_data)
            result['validation_report'] = validation_report

            if not validation_report.is_valid:
                logger.warning(
                    "Data validation issues found",
                    errors=validation_report.errors,
                    warnings=validation_report.warnings
                )

        # Step 5: Collect cache statistics
        if use_cache:
            result['cache_stats'] = cache_manager.get_global_stats()

        logger.info(
            "Data pipeline completed successfully",
            cell_count=len(cell_ids),
            cycle_rows=len(processed_data),
            quality_score=result['validation_report'].quality_score if result['validation_report'] else None
        )

    except Exception as e:
        logger.error("Data pipeline failed", error=str(e))
        result['errors'].append(str(e))

    return result


def validate_data_sources() -> Dict[str, Any]:
    """
    Validate that all data sources are properly configured and accessible.

    Returns:
        Dictionary with validation results for each data source
    """
    logger.info("Validating data sources")

    results = {
        'redash': validate_data_source_config(),
        'cache': _validate_cache_setup(),
        'overall_status': 'unknown'
    }

    # Determine overall status
    redash_ok = all(results['redash'].values())
    cache_ok = results['cache']['status'] == 'ok'

    if redash_ok and cache_ok:
        results['overall_status'] = 'ok'
    elif redash_ok or cache_ok:
        results['overall_status'] = 'partial'
    else:
        results['overall_status'] = 'failed'

    logger.info(
        "Data source validation completed",
        overall_status=results['overall_status'],
        redash_status=redash_ok,
        cache_status=cache_ok
    )

    return results


def _validate_cache_setup() -> Dict[str, Any]:
    """Validate cache configuration"""
    try:
        # Test cache operations
        test_cache = get_query_cache()
        test_key = "validation_test"
        test_data = {"test": "data"}

        # Test put/get
        test_cache.put(test_key, test_data, ttl=10)
        retrieved = test_cache.get(test_key)

        # Clean up
        test_cache.delete(test_key)

        success = retrieved == test_data

        return {
            'status': 'ok' if success else 'failed',
            'cache_count': len(cache_manager._caches),
            'test_passed': success
        }

    except Exception as e:
        logger.error("Cache validation failed", error=str(e))
        return {
            'status': 'failed',
            'error': str(e),
            'cache_count': 0,
            'test_passed': False
        }


def clear_all_caches():
    """Clear all data caches"""
    logger.info("Clearing all caches")

    # Clear query cache from loaders
    clear_query_cache()

    # Clear managed caches
    cache_manager.clear_all_caches()

    logger.info("All caches cleared")


def get_data_module_stats() -> Dict[str, Any]:
    """Get comprehensive statistics about the data module"""

    stats = {
        'cache_stats': cache_manager.get_global_stats(),
        'query_cache_stats': get_cache_stats(),
        'data_source_status': validate_data_sources(),
        'module_info': {
            'version': __version__,
            'available_caches': list(cache_manager._caches.keys()),
            'processors_available': True,
            'validators_available': True,
            'loaders_available': True
        }
    }

    return stats


# Convenience data loading functions
async def async_load_cells_data(
        cell_ids: List[int],
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Asynchronously load data for specific cells.

    Args:
        cell_ids: List of cell IDs to load
        use_cache: Whether to use caching
        progress_callback: Optional progress callback function

    Returns:
        Dictionary with loaded data and metadata
    """
    logger.info("Starting async cell data loading", cell_count=len(cell_ids))

    try:
        # Load cell metadata first
        if progress_callback:
            progress_callback(0.1, "Loading cell metadata")

        cell_data = load_initial_data()

        # Filter to requested cells
        available_cells = cell_data.filter(pl.col('cell_id').is_in(cell_ids))

        if progress_callback:
            progress_callback(0.3, "Loading cycle data")

        # Load cycle data
        cycle_data = await get_cycle_data(
            cell_ids,
            cell_metadata=available_cells,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback(1.0, "Complete")

        return {
            'cell_data': available_cells,
            'cycle_data': cycle_data,
            'success': True,
            'cell_count': len(available_cells),
            'cycle_count': len(cycle_data) if not cycle_data.is_empty() else 0
        }

    except Exception as e:
        logger.error("Async cell data loading failed", error=str(e))
        return {
            'cell_data': None,
            'cycle_data': None,
            'success': False,
            'error': str(e)
        }


def create_data_summary(data: Any) -> Dict[str, Any]:
    """Create a summary of any data object"""
    import polars as pl

    summary = {
        'type': type(data).__name__,
        'timestamp': datetime.now().isoformat()
    }

    if isinstance(data, pl.DataFrame):
        summary.update({
            'rows': len(data),
            'columns': len(data.columns),
            'memory_mb': data.estimated_size() / (1024 * 1024),
            'column_names': data.columns,
            'dtypes': {col: str(data[col].dtype) for col in data.columns},
            'null_counts': {col: data[col].null_count() for col in data.columns},
            'shape': data.shape
        })

        # Add basic statistics for numeric columns
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if numeric_cols:
            summary['numeric_stats'] = {}
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                try:
                    summary['numeric_stats'][col] = {
                        'min': data[col].min(),
                        'max': data[col].max(),
                        'mean': data[col].mean(),
                        'std': data[col].std()
                    }
                except Exception:
                    pass

    elif isinstance(data, dict):
        summary.update({
            'keys': list(data.keys()),
            'key_count': len(data)
        })

    elif isinstance(data, (list, tuple)):
        summary.update({
            'length': len(data),
            'element_types': list(set(type(item).__name__ for item in data[:10]))  # Sample first 10
        })

    return summary


# Module initialization
def initialize_data_module():
    """Initialize the data module with default settings"""
    logger.info("Initializing data module", version=__version__)

    # Validate data sources
    validation_results = validate_data_sources()

    if validation_results['overall_status'] == 'failed':
        logger.warning("Data source validation failed", results=validation_results)
    else:
        logger.info("Data module initialized successfully", status=validation_results['overall_status'])

    return validation_results


# Export all public functions and classes
__all__ = [
    # Main pipeline functions
    'load_and_process_data',
    'validate_data_sources',
    'clear_all_caches',
    'get_data_module_stats',
    'async_load_cells_data',
    'create_data_summary',
    'initialize_data_module',

    # From loaders
    'RedashClient',
    'get_redash_query_results',
    'load_initial_data',
    'get_cycle_data',
    'validate_data_source_config',

    # From processors
    'DataProcessor',
    'ProcessingConfig',
    'NormalizationMethod',
    'process_battery_data',
    'create_processing_config',
    'add_specific_capacity_energy',
    'calculate_coulombic_efficiency',
    'add_temperature_features',

    # From cache
    'DataCache',
    'CacheManager',
    'CacheStrategy',
    'cache_manager',
    'get_cell_cache',
    'get_cycle_cache',
    'get_query_cache',
    'get_processed_cache',
    'cache_result',
    'warm_cache_for_cells',

    # From validators
    'DataValidator',
    'ValidationRule',
    'ValidationResult',
    'DataQualityReport',
    'ValidationType',
    'Severity',
    'validate_battery_data',
    'validate_cell_metadata',
    'validate_cycle_data',
    'quick_data_check',
    'calculate_data_completeness',
    'calculate_data_consistency'
]

# Auto-initialize when module is imported
try:
    _init_results = initialize_data_module()
except Exception as e:
    logger.error("Failed to initialize data module", error=str(e))
    _init_results = None
# battery_dashboard/data/processors.py
"""
Data processing utilities for battery analytics dashboard.
Handles data transformation, normalization, and feature engineering.
"""
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import structlog
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

from ..core.exceptions import DataProcessingError, DataValidationError
from ..utils.decorators import timing_decorator
from ..utils.helpers import safe_divide, validate_numeric_column

logger = structlog.get_logger(__name__)


class NormalizationMethod(Enum):
    """Supported normalization methods"""
    FIRST_CYCLE = "first_cycle"
    MAX_VALUE = "max_value"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations"""
    add_normalizations: bool = True
    add_degradation_features: bool = True
    add_statistical_features: bool = True
    add_cycle_features: bool = True
    normalization_methods: List[NormalizationMethod] = None
    min_cycles_required: int = 20
    outlier_detection: bool = True
    outlier_threshold: float = 3.0

    def __post_init__(self):
        if self.normalization_methods is None:
            self.normalization_methods = [
                NormalizationMethod.FIRST_CYCLE,
                NormalizationMethod.PERCENTILE_95
            ]


class DataProcessor:
    """Main data processing class for battery analytics"""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)

    @timing_decorator
    def process_cycle_data(
            self,
            data: pl.DataFrame,
            cell_metadata: Optional[pl.DataFrame] = None,
            group_by_cell: bool = True
    ) -> pl.DataFrame:
        """
        Main processing function for cycle data.

        Args:
            data: Raw cycle data
            cell_metadata: Optional cell metadata for enrichment
            group_by_cell: Whether to apply processing per cell

        Returns:
            Processed DataFrame with added features
        """
        if data.is_empty():
            logger.warning("Empty DataFrame provided for processing")
            return data

        logger.info(
            "Starting cycle data processing",
            rows=len(data),
            columns=len(data.columns),
            config=self.config
        )

        processed_data = data.clone()

        try:
            # Basic data validation
            processed_data = self._validate_and_clean_data(processed_data)

            # Add basic cycle features
            if self.config.add_cycle_features:
                processed_data = self._add_cycle_features(processed_data)

            # Process by cell if specified
            if group_by_cell and 'cell_id' in processed_data.columns:
                processed_data = self._process_by_cell(processed_data)
            else:
                # Apply global processing
                processed_data = self._apply_processing_pipeline(processed_data)

            # Add statistical features across all data
            if self.config.add_statistical_features:
                processed_data = self._add_statistical_features(processed_data)

            # Merge with cell metadata if provided
            if cell_metadata is not None:
                processed_data = self._merge_with_metadata(processed_data, cell_metadata)

            # Final data quality checks
            processed_data = self._final_quality_checks(processed_data)

            logger.info(
                "Cycle data processing completed",
                original_rows=len(data),
                final_rows=len(processed_data),
                added_columns=len(processed_data.columns) - len(data.columns)
            )

            return processed_data

        except Exception as e:
            logger.error("Failed to process cycle data", error=str(e))
            raise DataProcessingError(f"Data processing failed: {str(e)}") from e

    def _validate_and_clean_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and clean input data"""
        logger.info("Validating and cleaning data")

        # Check for required columns
        required_columns = ['cell_id']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        # Remove rows with null cell_id
        initial_rows = len(data)
        data = data.filter(pl.col('cell_id').is_not_null())

        if len(data) < initial_rows:
            logger.warning(
                "Removed rows with null cell_id",
                removed_rows=initial_rows - len(data)
            )

        # Sort by cell_id and cycle number if available
        sort_columns = ['cell_id']
        if 'regular_cycle_number' in data.columns:
            sort_columns.append('regular_cycle_number')
        elif 'cycle_number' in data.columns:
            sort_columns.append('cycle_number')

        data = data.sort(sort_columns)

        return data

    def _add_cycle_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add basic cycle-level features"""
        logger.info("Adding cycle features")

        expressions = []

        # Add cycle age if not present
        if 'cycle_age_days' not in data.columns and 'test_date' in data.columns:
            expressions.append(
                (pl.col('test_date') - pl.col('test_date').min().over('cell_id'))
                .dt.total_days()
                .alias('cycle_age_days')
            )

        # Add cumulative cycle count
        if 'cumulative_cycles' not in data.columns and 'regular_cycle_number' in data.columns:
            expressions.append(
                pl.col('regular_cycle_number').alias('cumulative_cycles')
            )

        # Add cycle efficiency metrics
        capacity_cols = self._find_capacity_columns(data)
        for col in capacity_cols:
            if f'{col}_efficiency' not in data.columns:
                # Simple efficiency as ratio to theoretical maximum
                expressions.append(
                    (pl.col(col) / pl.col(col).max().over('cell_id')).alias(f'{col}_efficiency')
                )

        if expressions:
            data = data.with_columns(expressions)

        return data

    def _process_by_cell(self, data: pl.DataFrame) -> pl.DataFrame:
        """Apply processing pipeline to each cell group"""
        logger.info("Processing data by cell groups")

        cell_ids = data['cell_id'].unique().to_list()
        logger.info(f"Processing {len(cell_ids)} cells")

        processed_groups = []

        for cell_id in cell_ids:
            cell_data = data.filter(pl.col('cell_id') == cell_id)

            # Skip cells with insufficient data
            if len(cell_data) < self.config.min_cycles_required:
                logger.warning(
                    "Skipping cell with insufficient cycles",
                    cell_id=cell_id,
                    cycles=len(cell_data),
                    min_required=self.config.min_cycles_required
                )
                continue

            try:
                processed_cell = self._apply_processing_pipeline(cell_data)
                processed_groups.append(processed_cell)
            except Exception as e:
                logger.error(
                    "Failed to process cell",
                    cell_id=cell_id,
                    error=str(e)
                )
                # Include original data if processing fails
                processed_groups.append(cell_data)

        if not processed_groups:
            logger.warning("No cells processed successfully")
            return data

        return pl.concat(processed_groups)

    def _apply_processing_pipeline(self, data: pl.DataFrame) -> pl.DataFrame:
        """Apply the main processing pipeline to data"""

        # Add normalizations
        if self.config.add_normalizations:
            data = self._add_normalizations(data)

        # Add degradation features
        if self.config.add_degradation_features:
            data = self._add_degradation_features(data)

        # Detect and handle outliers
        if self.config.outlier_detection:
            data = self._handle_outliers(data)

        return data

    def _add_normalizations(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add various normalization columns"""
        logger.debug("Adding normalizations")

        # Find columns to normalize
        normalize_columns = self._find_normalizable_columns(data)

        if not normalize_columns:
            return data

        expressions = []

        for col in normalize_columns:
            for method in self.config.normalization_methods:
                norm_col_name = f"{col}_norm_{method.value}"

                if method == NormalizationMethod.FIRST_CYCLE:
                    # Use first non-null value
                    first_val = data.filter(pl.col(col).is_not_null())[col].first()
                    if first_val and first_val > 0:
                        expressions.append(
                            safe_divide(pl.col(col), first_val).alias(norm_col_name)
                        )

                elif method == NormalizationMethod.MAX_VALUE:
                    max_val = data[col].max()
                    if max_val and max_val > 0:
                        expressions.append(
                            safe_divide(pl.col(col), max_val).alias(norm_col_name)
                        )

                elif method == NormalizationMethod.PERCENTILE_95:
                    p95_val = data[col].quantile(0.95)
                    if p95_val and p95_val > 0:
                        expressions.append(
                            safe_divide(pl.col(col), p95_val).alias(norm_col_name)
                        )

                elif method == NormalizationMethod.PERCENTILE_99:
                    p99_val = data[col].quantile(0.99)
                    if p99_val and p99_val > 0:
                        expressions.append(
                            safe_divide(pl.col(col), p99_val).alias(norm_col_name)
                        )

                elif method == NormalizationMethod.Z_SCORE:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val and std_val > 0:
                        expressions.append(
                            ((pl.col(col) - mean_val) / std_val).alias(norm_col_name)
                        )

                elif method == NormalizationMethod.MIN_MAX:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val is not None and min_val is not None and max_val > min_val:
                        expressions.append(
                            ((pl.col(col) - min_val) / (max_val - min_val)).alias(norm_col_name)
                        )

        if expressions:
            data = data.with_columns(expressions)
            logger.debug(f"Added {len(expressions)} normalization columns")

        return data

    def _add_degradation_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add degradation-related features"""
        logger.debug("Adding degradation features")

        if 'regular_cycle_number' not in data.columns:
            return data

        capacity_cols = self._find_capacity_columns(data)
        expressions = []

        for col in capacity_cols:
            # Capacity retention (as fraction of first cycle)
            first_cycle_data = data.filter(pl.col('regular_cycle_number') == 1)
            if not first_cycle_data.is_empty():
                first_val = first_cycle_data[col].first()
                if first_val and first_val > 0:
                    expressions.append(
                        safe_divide(pl.col(col), first_val).alias(f"{col}_retention")
                    )

            # Capacity fade rate (derivative approximation)
            expressions.append(
                (pl.col(col).diff() / pl.col('regular_cycle_number').diff())
                .alias(f"{col}_fade_rate")
            )

            # Cumulative capacity loss
            if first_val and first_val > 0:
                expressions.append(
                    (first_val - pl.col(col)).alias(f"{col}_cumulative_loss")
                )

        # Add cycle-based features
        if len(data) > 1:
            # Cycles to 80% capacity
            for col in capacity_cols:
                first_cycle_data = data.filter(pl.col('regular_cycle_number') == 1)
                if not first_cycle_data.is_empty():
                    first_val = first_cycle_data[col].first()
                    if first_val and first_val > 0:
                        threshold = 0.8 * first_val
                        # Find first cycle where capacity drops below threshold
                        below_threshold = data.filter(pl.col(col) < threshold)
                        if not below_threshold.is_empty():
                            eol_cycle = below_threshold['regular_cycle_number'].min()
                            expressions.append(
                                pl.lit(eol_cycle).alias(f"{col}_cycles_to_80pct")
                            )

        if expressions:
            data = data.with_columns(expressions)
            logger.debug(f"Added {len(expressions)} degradation features")

        return data

    def _add_statistical_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add statistical features across the dataset"""
        logger.debug("Adding statistical features")

        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

        if not numeric_cols:
            return data

        # Add rolling statistics for key metrics
        window_sizes = [5, 10, 20]
        capacity_cols = self._find_capacity_columns(data)

        expressions = []

        for col in capacity_cols:
            for window in window_sizes:
                if len(data) >= window:
                    # Rolling mean
                    expressions.append(
                        pl.col(col).rolling_mean(window_size=window).alias(f"{col}_rolling_mean_{window}")
                    )
                    # Rolling std
                    expressions.append(
                        pl.col(col).rolling_std(window_size=window).alias(f"{col}_rolling_std_{window}")
                    )

        if expressions:
            data = data.with_columns(expressions)
            logger.debug(f"Added {len(expressions)} statistical features")

        return data

    def _handle_outliers(self, data: pl.DataFrame) -> pl.DataFrame:
        """Detect and handle outliers using IQR method"""
        logger.debug("Handling outliers")

        numeric_cols = self._find_capacity_columns(data)
        outlier_flags = []

        for col in numeric_cols:
            if validate_numeric_column(data, col):
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - self.config.outlier_threshold * iqr
                upper_bound = q3 + self.config.outlier_threshold * iqr

                outlier_flag = (
                        (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                ).alias(f"{col}_outlier")

                outlier_flags.append(outlier_flag)

        if outlier_flags:
            data = data.with_columns(outlier_flags)

            # Count outliers
            outlier_counts = {}
            for col in numeric_cols:
                outlier_col = f"{col}_outlier"
                if outlier_col in data.columns:
                    count = data.filter(pl.col(outlier_col) == True).height
                    outlier_counts[col] = count

            logger.info("Outlier detection completed", outlier_counts=outlier_counts)

        return data

    def _merge_with_metadata(self, data: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
        """Merge cycle data with cell metadata"""
        logger.debug("Merging with cell metadata")

        if 'cell_id' not in metadata.columns:
            logger.warning("No cell_id in metadata, skipping merge")
            return data

        # Find columns to merge (avoid duplicates)
        existing_cols = set(data.columns)
        metadata_cols = [col for col in metadata.columns if col != 'cell_id']
        unique_metadata_cols = [col for col in metadata_cols if col not in existing_cols]

        if not unique_metadata_cols:
            logger.info("No unique metadata columns to merge")
            return data

        # Perform the join
        try:
            cols_to_select = ['cell_id'] + unique_metadata_cols
            merged_data = data.join(
                metadata.select(cols_to_select),
                on='cell_id',
                how='left'
            )

            logger.info(
                "Merged with metadata",
                added_columns=len(unique_metadata_cols),
                final_columns=len(merged_data.columns)
            )

            return merged_data

        except Exception as e:
            logger.error("Failed to merge with metadata", error=str(e))
            return data

    def _final_quality_checks(self, data: pl.DataFrame) -> pl.DataFrame:
        """Perform final data quality checks"""
        logger.debug("Performing final quality checks")

        # Check for infinite values
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float64, pl.Float32]]

        for col in numeric_cols:
            inf_count = data.filter(pl.col(col).is_infinite()).height
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in {col}")
                # Replace infinities with null
                data = data.with_columns(
                    pl.when(pl.col(col).is_infinite())
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        # Log final statistics
        null_counts = {col: data[col].null_count() for col in data.columns}
        logger.info("Final data quality stats", null_counts=null_counts)

        return data

    def _find_normalizable_columns(self, data: pl.DataFrame) -> List[str]:
        """Find columns that should be normalized"""
        patterns = ['capacity', 'energy', 'voltage', 'current', 'power']
        normalizable = []

        for col in data.columns:
            if any(pattern in col.lower() for pattern in patterns):
                # Check if it's numeric
                if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    # Check if it has reasonable variance
                    if data[col].std() > 0:
                        normalizable.append(col)

        return normalizable

    def _find_capacity_columns(self, data: pl.DataFrame) -> List[str]:
        """Find capacity-related columns"""
        capacity_patterns = ['capacity', 'cap_']
        capacity_cols = []

        for col in data.columns:
            if any(pattern in col.lower() for pattern in capacity_patterns):
                if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    capacity_cols.append(col)

        return capacity_cols


# Specific processing functions
@timing_decorator
def add_specific_capacity_energy(data: pl.DataFrame) -> pl.DataFrame:
    """Add specific capacity and energy metrics (mAh/g, Wh/g)"""
    if 'total_active_mass_g' not in data.columns:
        logger.warning("No total_active_mass_g column found, skipping specific calculations")
        return data

    # Find capacity and energy columns
    capacity_cols = [col for col in data.columns if 'capacity' in col.lower() and '_specific_' not in col]
    energy_cols = [col for col in data.columns if 'energy' in col.lower() and '_specific_' not in col]

    expressions = []

    # Capacity: convert to mAh/g
    for col in capacity_cols:
        expressions.append(
            pl.when(pl.col('total_active_mass_g') > 0)
            .then(1000 * pl.col(col) / pl.col('total_active_mass_g'))
            .otherwise(None)
            .alias(f"{col}_specific_mAh_g")
        )

    # Energy: convert to Wh/g
    for col in energy_cols:
        expressions.append(
            pl.when(pl.col('total_active_mass_g') > 0)
            .then(pl.col(col) / pl.col('total_active_mass_g'))
            .otherwise(None)
            .alias(f"{col}_specific_Wh_g")
        )

    if expressions:
        data = data.with_columns(expressions)
        logger.info(f"Added {len(expressions)} specific capacity/energy columns")

    return data


@timing_decorator
def calculate_coulombic_efficiency(data: pl.DataFrame) -> pl.DataFrame:
    """Calculate coulombic efficiency if charge/discharge capacities are available"""
    charge_cols = [col for col in data.columns if 'charge' in col.lower() and 'capacity' in col.lower()]
    discharge_cols = [col for col in data.columns if 'discharge' in col.lower() and 'capacity' in col.lower()]

    if not charge_cols or not discharge_cols:
        logger.info("Charge/discharge capacity columns not found, skipping coulombic efficiency")
        return data

    expressions = []

    for charge_col in charge_cols:
        for discharge_col in discharge_cols:
            if 'charge' in charge_col and 'discharge' in discharge_col:
                eff_col_name = f"coulombic_efficiency_{charge_col}_{discharge_col}"
                expressions.append(
                    safe_divide(pl.col(discharge_col), pl.col(charge_col)).alias(eff_col_name)
                )

    if expressions:
        data = data.with_columns(expressions)
        logger.info(f"Added {len(expressions)} coulombic efficiency columns")

    return data


@timing_decorator
def add_temperature_features(data: pl.DataFrame) -> pl.DataFrame:
    """Add temperature-related features if temperature data is available"""
    temp_cols = [col for col in data.columns if 'temp' in col.lower()]

    if not temp_cols:
        return data

    expressions = []

    for temp_col in temp_cols:
        # Temperature statistics
        expressions.extend([
            pl.col(temp_col).rolling_mean(window_size=5).alias(f"{temp_col}_rolling_5"),
            pl.col(temp_col).rolling_std(window_size=5).alias(f"{temp_col}_std_5"),
            (pl.col(temp_col) - pl.col(temp_col).mean()).alias(f"{temp_col}_deviation"),
        ])

    if expressions:
        data = data.with_columns(expressions)
        logger.info(f"Added {len(expressions)} temperature features")

    return data


# Convenience functions
def process_battery_data(
        data: pl.DataFrame,
        cell_metadata: Optional[pl.DataFrame] = None,
        config: Optional[ProcessingConfig] = None
) -> pl.DataFrame:
    """
    Convenience function to process battery data with standard settings.

    Args:
        data: Raw cycle data
        cell_metadata: Optional cell metadata
        config: Processing configuration

    Returns:
        Processed DataFrame
    """
    processor = DataProcessor(config)
    return processor.process_cycle_data(data, cell_metadata)


def create_processing_config(
        normalization_methods: List[str] = None,
        min_cycles: int = 20,
        add_degradation: bool = True,
        add_specific: bool = True
) -> ProcessingConfig:
    """Create a processing configuration with common settings"""

    methods = []
    if normalization_methods:
        for method_str in normalization_methods:
            try:
                methods.append(NormalizationMethod(method_str))
            except ValueError:
                logger.warning(f"Unknown normalization method: {method_str}")

    return ProcessingConfig(
        normalization_methods=methods or [NormalizationMethod.FIRST_CYCLE, NormalizationMethod.PERCENTILE_95],
        min_cycles_required=min_cycles,
        add_degradation_features=add_degradation,
        add_normalizations=True,
        add_statistical_features=add_specific
    )
# battery_dashboard/utils/helpers.py
import polars as pl
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import re
from datetime import datetime, timedelta


def safe_divide(numerator: Union[float, pl.Expr], denominator: Union[float, pl.Expr],
                default: float = 0.0) -> Union[float, pl.Expr]:
    """Safely divide two numbers, returning default if denominator is zero"""
    if isinstance(numerator, pl.Expr) or isinstance(denominator, pl.Expr):
        # Polars expression version
        return pl.when(denominator != 0).then(numerator / denominator).otherwise(default)
    else:
        # Regular number version
        return numerator / denominator if denominator != 0 else default


def normalize_column_names(df: Union[pl.DataFrame, pd.DataFrame]) -> Union[pl.DataFrame, pd.DataFrame]:
    """Normalize column names to snake_case"""

    def to_snake_case(name: str) -> str:
        # Replace spaces and special characters with underscores
        name = re.sub(r'[-\s]+', '_', name)
        # Insert underscore before uppercase letters
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        # Convert to lowercase and remove multiple underscores
        name = re.sub(r'_+', '_', name.lower())
        # Remove leading/trailing underscores
        return name.strip('_')

    if isinstance(df, pl.DataFrame):
        return df.rename({col: to_snake_case(col) for col in df.columns})
    else:
        return df.rename(columns={col: to_snake_case(col) for col in df.columns})


def detect_outliers(data: Union[pl.Series, np.ndarray, List], method: str = "iqr") -> np.ndarray:
    """Detect outliers using various methods"""
    if isinstance(data, pl.Series):
        values = data.to_numpy()
    elif isinstance(data, list):
        values = np.array(data)
    else:
        values = data

    if method == "iqr":
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (values < lower_bound) | (values > upper_bound)

    elif method == "zscore":
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        return z_scores > 3

    elif method == "modified_zscore":
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        modified_z_scores = 0.6745 * (values - median) / mad
        return np.abs(modified_z_scores) > 3.5

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def calculate_battery_metrics(cycle_data: pl.DataFrame) -> Dict[str, Any]:
    """Calculate common battery performance metrics"""
    metrics = {}

    if cycle_data.is_empty():
        return metrics

    # Basic statistics
    if "discharge_capacity" in cycle_data.columns:
        capacity_data = cycle_data["discharge_capacity"].drop_nulls()
        if len(capacity_data) > 0:
            metrics["capacity_mean"] = capacity_data.mean()
            metrics["capacity_std"] = capacity_data.std()
            metrics["capacity_min"] = capacity_data.min()
            metrics["capacity_max"] = capacity_data.max()

            # Capacity retention (assuming first cycle is reference)
            first_capacity = capacity_data.first()
            last_capacity = capacity_data.last()
            if first_capacity and first_capacity > 0:
                metrics["capacity_retention"] = (last_capacity / first_capacity) * 100

    # Coulombic efficiency
    if "coulombic_efficiency" in cycle_data.columns:
        ce_data = cycle_data["coulombic_efficiency"].drop_nulls()
        if len(ce_data) > 0:
            metrics["coulombic_efficiency_mean"] = ce_data.mean()
            metrics["coulombic_efficiency_std"] = ce_data.std()

    # Cycle count
    if "regular_cycle_number" in cycle_data.columns:
        cycle_numbers = cycle_data["regular_cycle_number"].drop_nulls()
        if len(cycle_numbers) > 0:
            metrics["total_cycles"] = cycle_numbers.max()
            metrics["cycle_range"] = f"{cycle_numbers.min()}-{cycle_numbers.max()}"

    return metrics


def format_large_number(number: Union[int, float], precision: int = 1) -> str:
    """Format large numbers with K, M, B suffixes"""
    if number is None:
        return "N/A"

    if abs(number) >= 1e9:
        return f"{number / 1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number / 1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number / 1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def create_time_bins(timestamps: Union[pl.Series, pd.Series, List],
                     bin_size: str = "1h") -> Union[pl.Series, pd.Series]:
    """Create time bins for grouping time series data"""
    if isinstance(timestamps, list):
        timestamps = pd.Series(timestamps)

    if isinstance(timestamps, pl.Series):
        # Convert to pandas for easier time operations
        pd_timestamps = timestamps.to_pandas()
        binned = pd.cut(pd_timestamps, bins=pd.date_range(
            start=pd_timestamps.min(),
            end=pd_timestamps.max(),
            freq=bin_size
        ))
        return pl.from_pandas(pd.Series(binned))
    else:
        # Pandas series
        return pd.cut(timestamps, bins=pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq=bin_size
        ))


def validate_dataframe_schema(df: Union[pl.DataFrame, pd.DataFrame],
                              required_columns: List[str],
                              optional_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate dataframe schema and return validation results"""
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {}
    }

    # Check required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        results["valid"] = False
        results["errors"].append(f"Missing required columns: {missing_required}")

    # Check for empty dataframe
    if len(df) == 0:
        results["valid"] = False
        results["errors"].append("DataFrame is empty")

    # Check for duplicate columns
    if len(set(df.columns)) != len(df.columns):
        duplicates = [col for col in df.columns if df.columns.count(col) > 1]
        results["warnings"].append(f"Duplicate columns found: {set(duplicates)}")

    # Summary statistics
    results["summary"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "required_columns_present": len(required_columns) - len(missing_required),
        "memory_usage": df.estimated_size("mb") if isinstance(df, pl.DataFrame) else df.memory_usage(deep=True).sum()
    }

    return results


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between common battery-related units"""

    # Energy conversions (base: Wh)
    energy_conversions = {
        "wh": 1.0,
        "kwh": 1000.0,
        "mwh": 1000000.0,
        "j": 1 / 3600.0,
        "kj": 1000 / 3600.0,
        "mj": 1000000 / 3600.0
    }

    # Capacity conversions (base: Ah)
    capacity_conversions = {
        "ah": 1.0,
        "mah": 0.001,
        "c": 3600.0,  # Coulombs
        "mc": 0.0036  # MilliCoulombs
    }

    # Power conversions (base: W)
    power_conversions = {
        "w": 1.0,
        "kw": 1000.0,
        "mw": 1000000.0,
        "hp": 745.7
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Determine conversion type
    if from_unit in energy_conversions and to_unit in energy_conversions:
        return value * energy_conversions[from_unit] / energy_conversions[to_unit]
    elif from_unit in capacity_conversions and to_unit in capacity_conversions:
        return value * capacity_conversions[from_unit] / capacity_conversions[to_unit]
    elif from_unit in power_conversions and to_unit in power_conversions:
        return value * power_conversions[from_unit] / power_conversions[to_unit]
    else:
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
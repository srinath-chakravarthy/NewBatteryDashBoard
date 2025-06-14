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


def validate_numeric_columns(
        df: Union[pl.DataFrame, pd.DataFrame],
        columns: List[str],
        allow_null: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        check_infinite: bool = True,
        check_outliers: bool = False,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Validate numeric columns in a DataFrame for battery analytics.

    Args:
        df: Input DataFrame (Polars or Pandas)
        columns: List of column names to validate
        allow_null: Whether to allow null/NaN values
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        check_infinite: Whether to check for infinite values
        check_outliers: Whether to detect outliers
        outlier_method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        outlier_threshold: Threshold for outlier detection

    Returns:
        Dict with validation results including:
        - valid: Overall validation status
        - errors: List of validation errors
        - warnings: List of validation warnings
        - column_stats: Statistics for each validated column
        - outlier_counts: Number of outliers per column (if check_outliers=True)
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "column_stats": {},
        "outlier_counts": {}
    }

    # Check if columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        results["valid"] = False
        results["errors"].append(f"Missing columns: {missing_columns}")
        return results

    for col in columns:
        col_results = {
            "data_type": None,
            "null_count": 0,
            "inf_count": 0,
            "total_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None
        }

        # Handle Polars vs Pandas differences
        if isinstance(df, pl.DataFrame):
            col_data = df[col]
            col_results["total_count"] = len(col_data)
            col_results["data_type"] = str(col_data.dtype)

            # Check if column is numeric
            if not col_data.dtype.is_numeric():
                results["valid"] = False
                results["errors"].append(f"Column '{col}' is not numeric (type: {col_data.dtype})")
                continue

            # Count nulls
            col_results["null_count"] = col_data.null_count()

            # Convert to numpy for calculations
            col_array = col_data.drop_nulls().to_numpy()

        else:  # pandas DataFrame
            col_data = df[col]
            col_results["total_count"] = len(col_data)
            col_results["data_type"] = str(col_data.dtype)

            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(col_data):
                results["valid"] = False
                results["errors"].append(f"Column '{col}' is not numeric (type: {col_data.dtype})")
                continue

            # Count nulls
            col_results["null_count"] = col_data.isnull().sum()

            # Convert to numpy for calculations
            col_array = col_data.dropna().values

        # Check for null values
        if not allow_null and col_results["null_count"] > 0:
            results["valid"] = False
            results["errors"].append(f"Column '{col}' contains {col_results['null_count']} null values")
        elif col_results["null_count"] > 0:
            results["warnings"].append(f"Column '{col}' contains {col_results['null_count']} null values")

        # Skip further validation if no non-null data
        if len(col_array) == 0:
            results["warnings"].append(f"Column '{col}' has no non-null values")
            results["column_stats"][col] = col_results
            continue

        # Check for infinite values
        if check_infinite:
            inf_mask = np.isinf(col_array)
            col_results["inf_count"] = np.sum(inf_mask)

            if col_results["inf_count"] > 0:
                results["valid"] = False
                results["errors"].append(f"Column '{col}' contains {col_results['inf_count']} infinite values")
                # Remove infinite values for further calculations
                col_array = col_array[~inf_mask]

        # Skip further validation if no finite data
        if len(col_array) == 0:
            results["warnings"].append(f"Column '{col}' has no finite values")
            results["column_stats"][col] = col_results
            continue

        # Calculate basic statistics
        try:
            col_results["min"] = float(np.min(col_array))
            col_results["max"] = float(np.max(col_array))
            col_results["mean"] = float(np.mean(col_array))
            col_results["std"] = float(np.std(col_array))
        except Exception as e:
            results["warnings"].append(f"Could not calculate statistics for column '{col}': {str(e)}")

        # Check value ranges
        if min_value is not None and col_results["min"] < min_value:
            results["valid"] = False
            results["errors"].append(f"Column '{col}' has values below minimum {min_value} (min: {col_results['min']})")

        if max_value is not None and col_results["max"] > max_value:
            results["valid"] = False
            results["errors"].append(f"Column '{col}' has values above maximum {max_value} (max: {col_results['max']})")

        # Outlier detection
        if check_outliers and len(col_array) > 3:  # Need at least 4 values for outlier detection
            try:
                outlier_mask = detect_outliers(col_array, method=outlier_method)
                outlier_count = np.sum(outlier_mask)
                results["outlier_counts"][col] = int(outlier_count)

                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(col_array)) * 100
                    if outlier_percentage > 10:  # More than 10% outliers
                        results["warnings"].append(
                            f"Column '{col}' has {outlier_count} outliers ({outlier_percentage:.1f}%)"
                        )
                    else:
                        results["warnings"].append(
                            f"Column '{col}' has {outlier_count} outliers ({outlier_percentage:.1f}%)"
                        )
            except Exception as e:
                results["warnings"].append(f"Could not detect outliers for column '{col}': {str(e)}")

        # Battery-specific validations
        if col.lower() in ['discharge_capacity', 'charge_capacity', 'capacity']:
            if col_results["min"] is not None and col_results["min"] < 0:
                results["warnings"].append(f"Column '{col}' has negative capacity values")

        if col.lower() in ['coulombic_efficiency', 'efficiency']:
            if col_results["max"] is not None and col_results["max"] > 100:
                results["warnings"].append(f"Column '{col}' has efficiency values > 100%")
            if col_results["min"] is not None and col_results["min"] < 0:
                results["warnings"].append(f"Column '{col}' has negative efficiency values")

        if col.lower() in ['voltage', 'cell_voltage']:
            if col_results["min"] is not None and col_results["min"] < 0:
                results["warnings"].append(f"Column '{col}' has negative voltage values")
            if col_results["max"] is not None and col_results["max"] > 5:
                results["warnings"].append(f"Column '{col}' has unusually high voltage values (>{5}V)")

        if col.lower() in ['temperature']:
            if col_results["min"] is not None and col_results["min"] < -50:
                results["warnings"].append(f"Column '{col}' has unusually low temperature values")
            if col_results["max"] is not None and col_results["max"] > 100:
                results["warnings"].append(f"Column '{col}' has unusually high temperature values")

        results["column_stats"][col] = col_results

    return results


def validate_battery_cycle_data(df: Union[pl.DataFrame, pd.DataFrame]) -> Dict[str, Any]:
    """
    Specialized validation function for battery cycle data.

    Args:
        df: DataFrame containing battery cycle data

    Returns:
        Dict with validation results
    """
    # Common battery cycle data columns
    numeric_columns = []

    # Check which columns exist and add to validation list
    potential_columns = {
        'discharge_capacity': {'min_value': 0, 'max_value': 10},  # Ah
        'charge_capacity': {'min_value': 0, 'max_value': 10},  # Ah
        'coulombic_efficiency': {'min_value': 0, 'max_value': 101},  # %
        'cell_voltage': {'min_value': 0, 'max_value': 5},  # V
        'temperature': {'min_value': -50, 'max_value': 100},  # °C
        'regular_cycle_number': {'min_value': 0, 'max_value': 10000},
        'cycle_time': {'min_value': 0},  # seconds/hours
        'energy': {'min_value': 0, 'max_value': 50},  # Wh
        'resistance': {'min_value': 0, 'max_value': 1000}  # Ohms
    }

    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "column_validations": {}
    }

    for col_name, constraints in potential_columns.items():
        if col_name in df.columns:
            numeric_columns.append(col_name)

            # Validate individual column
            col_validation = validate_numeric_columns(
                df,
                [col_name],
                allow_null=True,
                check_outliers=True,
                **constraints
            )

            validation_results["column_validations"][col_name] = col_validation

            # Aggregate results
            if not col_validation["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(col_validation["errors"])

            validation_results["warnings"].extend(col_validation["warnings"])

    # Additional battery-specific validations
    if 'discharge_capacity' in df.columns and 'charge_capacity' in df.columns:
        # Check if charge capacity is generally >= discharge capacity
        if isinstance(df, pl.DataFrame):
            charge_discharge_ratio = (df['charge_capacity'] / df['discharge_capacity']).mean()
        else:
            charge_discharge_ratio = (df['charge_capacity'] / df['discharge_capacity']).mean()

        if charge_discharge_ratio < 0.95:
            validation_results["warnings"].append(
                f"Average charge/discharge ratio is low: {charge_discharge_ratio:.3f}"
            )

    return validation_results
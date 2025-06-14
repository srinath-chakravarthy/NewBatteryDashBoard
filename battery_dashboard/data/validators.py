# battery_dashboard/data/validators.py
"""
Data validation utilities for battery analytics dashboard.
Provides validation rules, data quality checks, and schema validation.
"""
import polars as pl
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import structlog
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
import numpy as np

from ..core.exceptions import DataValidationError
from ..utils.decorators import timing_decorator

logger = structlog.get_logger(__name__)


class ValidationType(Enum):
    """Types of validation checks"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    UNIQUENESS = "uniqueness"
    REFERENTIAL = "referential"


class Severity(Enum):
    """Validation error severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationRule:
    """Individual validation rule"""
    column: str
    validation_type: ValidationType
    severity: Severity = Severity.ERROR
    parameters: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    custom_func: Optional[Callable] = None

    def __post_init__(self):
        if self.message is None:
            self.message = f"Validation failed for column '{self.column}'"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule: ValidationRule
    passed: bool
    failed_rows: Optional[List[int]] = None
    error_count: int = 0
    details: Optional[str] = None

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical validation failure"""
        return not self.passed and self.rule.severity == Severity.ERROR


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    total_rows: int
    total_columns: int
    validation_results: List[ValidationResult]
    passed_validations: int = 0
    failed_validations: int = 0
    warnings: int = 0
    errors: int = 0
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self._calculate_summary()

    def _calculate_summary(self):
        """Calculate summary statistics"""
        for result in self.validation_results:
            if result.passed:
                self.passed_validations += 1
            else:
                self.failed_validations += 1
                if result.rule.severity == Severity.ERROR:
                    self.errors += 1
                elif result.rule.severity == Severity.WARNING:
                    self.warnings += 1

        # Calculate quality score (0-1)
        total_validations = len(self.validation_results)
        if total_validations > 0:
            # Weight errors more heavily than warnings
            error_weight = 1.0
            warning_weight = 0.5

            weighted_failures = (self.errors * error_weight) + (self.warnings * warning_weight)
            max_possible_score = total_validations * error_weight

            self.quality_score = max(0.0, 1.0 - (weighted_failures / max_possible_score))
        else:
            self.quality_score = 1.0

    @property
    def is_valid(self) -> bool:
        """Check if data passes all critical validations"""
        return self.errors == 0

    def get_critical_issues(self) -> List[ValidationResult]:
        """Get only critical validation failures"""
        return [result for result in self.validation_results if result.is_critical]


class DataValidator:
    """Main data validation class"""

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._standard_rules_loaded = False

    def add_rule(self, rule: ValidationRule) -> 'DataValidator':
        """Add a validation rule"""
        self.rules.append(rule)
        return self

    def add_required_column(self, column: str, severity: Severity = Severity.ERROR) -> 'DataValidator':
        """Add required column validation"""
        rule = ValidationRule(
            column=column,
            validation_type=ValidationType.REQUIRED,
            severity=severity,
            message=f"Required column '{column}' is missing"
        )
        return self.add_rule(rule)

    def add_type_check(
            self,
            column: str,
            expected_type: Union[pl.DataType, List[pl.DataType]],
            severity: Severity = Severity.ERROR
    ) -> 'DataValidator':
        """Add data type validation"""
        rule = ValidationRule(
            column=column,
            validation_type=ValidationType.TYPE_CHECK,
            severity=severity,
            parameters={'expected_type': expected_type},
            message=f"Column '{column}' has incorrect data type"
        )
        return self.add_rule(rule)

    def add_range_check(
            self,
            column: str,
            min_val: Optional[float] = None,
            max_val: Optional[float] = None,
            severity: Severity = Severity.WARNING
    ) -> 'DataValidator':
        """Add range validation"""
        rule = ValidationRule(
            column=column,
            validation_type=ValidationType.RANGE,
            severity=severity,
            parameters={'min_val': min_val, 'max_val': max_val},
            message=f"Column '{column}' values outside expected range"
        )
        return self.add_rule(rule)

    def add_pattern_check(
            self,
            column: str,
            pattern: str,
            severity: Severity = Severity.ERROR
    ) -> 'DataValidator':
        """Add regex pattern validation"""
        rule = ValidationRule(
            column=column,
            validation_type=ValidationType.PATTERN,
            severity=severity,
            parameters={'pattern': pattern},
            message=f"Column '{column}' values don't match required pattern"
        )
        return self.add_rule(rule)

    def add_custom_validation(
            self,
            column: str,
            func: Callable,
            severity: Severity = Severity.ERROR,
            message: Optional[str] = None
    ) -> 'DataValidator':
        """Add custom validation function"""
        rule = ValidationRule(
            column=column,
            validation_type=ValidationType.CUSTOM,
            severity=severity,
            custom_func=func,
            message=message or f"Custom validation failed for column '{column}'"
        )
        return self.add_rule(rule)

    def load_battery_standards(self) -> 'DataValidator':
        """Load standard validation rules for battery data"""
        if self._standard_rules_loaded:
            return self

        # Required columns for battery data
        self.add_required_column('cell_id')

        # Cell ID should be positive integer
        self.add_range_check('cell_id', min_val=1)

        # Cycle numbers should be positive
        for col in ['regular_cycle_number', 'cycle_number']:
            self.add_custom_validation(
                col,
                lambda df, col=col: self._validate_cycle_numbers(df, col),
                severity=Severity.WARNING,
                message=f"Cycle numbers in '{col}' should be sequential and positive"
            )

        # Capacity values should be positive
        capacity_patterns = ['capacity', 'cap_']
        self.add_custom_validation(
            'capacity_columns',
            lambda df: self._validate_capacity_values(df, capacity_patterns),
            severity=Severity.WARNING,
            message="Capacity values should be positive"
        )

        # Energy values should be positive
        energy_patterns = ['energy']
        self.add_custom_validation(
            'energy_columns',
            lambda df: self._validate_energy_values(df, energy_patterns),
            severity=Severity.WARNING,
            message="Energy values should be positive"
        )

        # Voltage ranges (typical Li-ion: 2.0-4.5V)
        voltage_patterns = ['voltage', 'volt']
        self.add_custom_validation(
            'voltage_columns',
            lambda df: self._validate_voltage_ranges(df, voltage_patterns, 1.0, 5.0),
            severity=Severity.WARNING,
            message="Voltage values outside typical Li-ion range (1.0-5.0V)"
        )

        # Temperature ranges (-40°C to 100°C)
        temp_patterns = ['temp', 'temperature']
        self.add_custom_validation(
            'temperature_columns',
            lambda df: self._validate_temperature_ranges(df, temp_patterns, -40, 100),
            severity=Severity.WARNING,
            message="Temperature values outside reasonable range (-40°C to 100°C)"
        )

        # Coulombic efficiency should be between 0 and 1.5 (allowing for some measurement error)
        self.add_custom_validation(
            'coulombic_efficiency',
            lambda df: self._validate_efficiency_values(df),
            severity=Severity.INFO,
            message="Coulombic efficiency values outside expected range (0-1.5)"
        )

        self._standard_rules_loaded = True
        logger.info("Loaded standard battery validation rules", rule_count=len(self.rules))
        return self

    @timing_decorator
    def validate(self, data: pl.DataFrame) -> DataQualityReport:
        """
        Validate DataFrame against all rules.

        Args:
            data: DataFrame to validate

        Returns:
            DataQualityReport with validation results
        """
        logger.info("Starting data validation", rows=len(data), columns=len(data.columns))

        if data.is_empty():
            logger.warning("Empty DataFrame provided for validation")
            return DataQualityReport(
                total_rows=0,
                total_columns=0,
                validation_results=[]
            )

        results = []

        for rule in self.rules:
            try:
                result = self._apply_rule(data, rule)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Validation rule failed",
                    rule=rule.column,
                    type=rule.validation_type.value,
                    error=str(e)
                )
                # Create a failed result for the exception
                results.append(ValidationResult(
                    rule=rule,
                    passed=False,
                    details=f"Validation rule failed with error: {str(e)}"
                ))

        report = DataQualityReport(
            total_rows=len(data),
            total_columns=len(data.columns),
            validation_results=results
        )

        logger.info(
            "Data validation completed",
            quality_score=report.quality_score,
            errors=report.errors,
            warnings=report.warnings,
            passed=report.passed_validations
        )

        return report

    def _apply_rule(self, data: pl.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply a single validation rule"""

        if rule.validation_type == ValidationType.REQUIRED:
            return self._validate_required_column(data, rule)

        elif rule.validation_type == ValidationType.TYPE_CHECK:
            return self._validate_type(data, rule)

        elif rule.validation_type == ValidationType.RANGE:
            return self._validate_range(data, rule)

        elif rule.validation_type == ValidationType.PATTERN:
            return self._validate_pattern(data, rule)

        elif rule.validation_type == ValidationType.CUSTOM:
            return self._validate_custom(data, rule)

        else:
            raise DataValidationError(f"Unknown validation type: {rule.validation_type}")

    def _validate_required_column(self, data: pl.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate that required column exists"""
        passed = rule.column in data.columns
        return ValidationResult(
            rule=rule,
            passed=passed,
            details=f"Column exists: {passed}"
        )

    def _validate_type(self, data: pl.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate column data type"""
        if rule.column not in data.columns:
            return ValidationResult(
                rule=rule,
                passed=False,
                details="Column does not exist"
            )

        expected_type = rule.parameters['expected_type']
        actual_type = data[rule.column].dtype

        if isinstance(expected_type, list):
            passed = actual_type in expected_type
        else:
            passed = actual_type == expected_type

        return ValidationResult(
            rule=rule,
            passed=passed,
            details=f"Expected: {expected_type}, Actual: {actual_type}"
        )

    def _validate_range(self, data: pl.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate value ranges"""
        if rule.column not in data.columns:
            return ValidationResult(
                rule=rule,
                passed=False,
                details="Column does not exist"
            )

        min_val = rule.parameters.get('min_val')
        max_val = rule.parameters.get('max_val')

        column_data = data[rule.column]

        # Filter out null values
        non_null_data = column_data.filter(column_data.is_not_null())

        if non_null_data.is_empty():
            return ValidationResult(
                rule=rule,
                passed=True,
                details="No non-null values to validate"
            )

        failed_conditions = []

        if min_val is not None:
            below_min = non_null_data.filter(non_null_data < min_val)
            if not below_min.is_empty():
                failed_conditions.append(f"{len(below_min)} values below {min_val}")

        if max_val is not None:
            above_max = non_null_data.filter(non_null_data > max_val)
            if not above_max.is_empty():
                failed_conditions.append(f"{len(above_max)} values above {max_val}")

        passed = len(failed_conditions) == 0
        details = "; ".join(failed_conditions) if failed_conditions else "All values in range"

        return ValidationResult(
            rule=rule,
            passed=passed,
            error_count=len(failed_conditions),
            details=details
        )

    def _validate_pattern(self, data: pl.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate regex pattern"""
        if rule.column not in data.columns:
            return ValidationResult(
                rule=rule,
                passed=False,
                details="Column does not exist"
            )

        pattern = rule.parameters['pattern']
        column_data = data[rule.column]

        # Convert to string and check pattern
        try:
            string_data = column_data.cast(pl.Utf8)
            matches = string_data.str.contains(pattern)
            non_matching = string_data.filter(~matches)

            passed = non_matching.is_empty()
            error_count = len(non_matching) if not passed else 0

            return ValidationResult(
                rule=rule,
                passed=passed,
                error_count=error_count,
                details=f"Pattern matches: {len(string_data) - error_count}/{len(string_data)}"
            )
        except Exception as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                details=f"Pattern validation failed: {str(e)}"
            )

    def _validate_custom(self, data: pl.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Apply custom validation function"""
        if rule.custom_func is None:
            return ValidationResult(
                rule=rule,
                passed=False,
                details="No custom function provided"
            )

        try:
            result = rule.custom_func(data)

            if isinstance(result, bool):
                return ValidationResult(
                    rule=rule,
                    passed=result,
                    details="Custom validation result"
                )
            elif isinstance(result, dict):
                return ValidationResult(
                    rule=rule,
                    passed=result.get('passed', False),
                    error_count=result.get('error_count', 0),
                    details=result.get('details', 'Custom validation')
                )
            else:
                return ValidationResult(
                    rule=rule,
                    passed=False,
                    details=f"Invalid custom function return type: {type(result)}"
                )
        except Exception as e:
            logger.error("Custom validation function failed", error=str(e))
            return ValidationResult(
                rule=rule,
                passed=False,
                details=f"Custom validation failed: {str(e)}"
            )

    # Custom validation functions for battery data
    def _validate_cycle_numbers(self, data: pl.DataFrame, column: str) -> Dict[str, Any]:
        """Validate cycle number sequences"""
        if column not in data.columns:
            return {'passed': True, 'details': f'Column {column} not found'}

        cycle_data = data[column].filter(data[column].is_not_null())

        if cycle_data.is_empty():
            return {'passed': True, 'details': 'No cycle data to validate'}

        issues = []

        # Check for positive values
        negative_cycles = cycle_data.filter(cycle_data <= 0)
        if not negative_cycles.is_empty():
            issues.append(f"{len(negative_cycles)} non-positive cycle numbers")

        # Check for reasonable sequence (group by cell_id if available)
        if 'cell_id' in data.columns:
            for cell_id in data['cell_id'].unique():
                cell_cycles = data.filter(pl.col('cell_id') == cell_id)[column]
                cell_cycles = cell_cycles.filter(cell_cycles.is_not_null()).sort()

                if len(cell_cycles) > 1:
                    # Check for large gaps
                    diffs = cell_cycles.diff().filter(pl.col(column).is_not_null())
                    large_gaps = diffs.filter(diffs > 100)  # Arbitrary threshold
                    if not large_gaps.is_empty():
                        issues.append(f"Cell {cell_id}: {len(large_gaps)} large cycle number gaps")

        passed = len(issues) == 0
        return {
            'passed': passed,
            'error_count': len(issues),
            'details': '; '.join(issues) if issues else 'Cycle numbers look valid'
        }

    def _validate_capacity_values(self, data: pl.DataFrame, patterns: List[str]) -> Dict[str, Any]:
        """Validate capacity values are positive and reasonable"""
        capacity_cols = [col for col in data.columns
                         if any(pattern.lower() in col.lower() for pattern in patterns)]

        if not capacity_cols:
            return {'passed': True, 'details': 'No capacity columns found'}

        issues = []

        for col in capacity_cols:
            col_data = data[col].filter(data[col].is_not_null())

            if col_data.is_empty():
                continue

            # Check for negative values
            negative_vals = col_data.filter(col_data < 0)
            if not negative_vals.is_empty():
                issues.append(f"{col}: {len(negative_vals)} negative values")

            # Check for unreasonably large values (>1000 Ah)
            large_vals = col_data.filter(col_data > 1000)
            if not large_vals.is_empty():
                issues.append(f"{col}: {len(large_vals)} unreasonably large values (>1000)")

            # Check for zero values (might be valid but worth noting)
            zero_vals = col_data.filter(col_data == 0)
            if not zero_vals.is_empty() and len(zero_vals) > len(col_data) * 0.1:
                issues.append(f"{col}: {len(zero_vals)} zero values ({len(zero_vals) / len(col_data) * 100:.1f}%)")

        passed = len(issues) == 0
        return {
            'passed': passed,
            'error_count': len(issues),
            'details': '; '.join(issues) if issues else f'Capacity values valid in {len(capacity_cols)} columns'
        }

    def _validate_energy_values(self, data: pl.DataFrame, patterns: List[str]) -> Dict[str, Any]:
        """Validate energy values are positive and reasonable"""
        energy_cols = [col for col in data.columns
                       if any(pattern.lower() in col.lower() for pattern in patterns)]

        if not energy_cols:
            return {'passed': True, 'details': 'No energy columns found'}

        issues = []

        for col in energy_cols:
            col_data = data[col].filter(data[col].is_not_null())

            if col_data.is_empty():
                continue

            # Check for negative values
            negative_vals = col_data.filter(col_data < 0)
            if not negative_vals.is_empty():
                issues.append(f"{col}: {len(negative_vals)} negative values")

            # Check for unreasonably large values (>10000 Wh)
            large_vals = col_data.filter(col_data > 10000)
            if not large_vals.is_empty():
                issues.append(f"{col}: {len(large_vals)} unreasonably large values (>10000)")

        passed = len(issues) == 0
        return {
            'passed': passed,
            'error_count': len(issues),
            'details': '; '.join(issues) if issues else f'Energy values valid in {len(energy_cols)} columns'
        }

    def _validate_voltage_ranges(self, data: pl.DataFrame, patterns: List[str], min_v: float, max_v: float) -> Dict[
        str, Any]:
        """Validate voltage values are within reasonable ranges"""
        voltage_cols = [col for col in data.columns
                        if any(pattern.lower() in col.lower() for pattern in patterns)]

        if not voltage_cols:
            return {'passed': True, 'details': 'No voltage columns found'}

        issues = []

        for col in voltage_cols:
            col_data = data[col].filter(data[col].is_not_null())

            if col_data.is_empty():
                continue

            # Check for values outside range
            below_min = col_data.filter(col_data < min_v)
            above_max = col_data.filter(col_data > max_v)

            if not below_min.is_empty():
                issues.append(f"{col}: {len(below_min)} values below {min_v}V")

            if not above_max.is_empty():
                issues.append(f"{col}: {len(above_max)} values above {max_v}V")

        passed = len(issues) == 0
        return {
            'passed': passed,
            'error_count': len(issues),
            'details': '; '.join(issues) if issues else f'Voltage values valid in {len(voltage_cols)} columns'
        }

    def _validate_temperature_ranges(self, data: pl.DataFrame, patterns: List[str], min_t: float, max_t: float) -> Dict[
        str, Any]:
        """Validate temperature values are within reasonable ranges"""
        temp_cols = [col for col in data.columns
                     if any(pattern.lower() in col.lower() for pattern in patterns)]

        if not temp_cols:
            return {'passed': True, 'details': 'No temperature columns found'}

        issues = []

        for col in temp_cols:
            col_data = data[col].filter(data[col].is_not_null())

            if col_data.is_empty():
                continue

            # Check for values outside range
            below_min = col_data.filter(col_data < min_t)
            above_max = col_data.filter(col_data > max_t)

            if not below_min.is_empty():
                issues.append(f"{col}: {len(below_min)} values below {min_t}°C")

            if not above_max.is_empty():
                issues.append(f"{col}: {len(above_max)} values above {max_t}°C")

        passed = len(issues) == 0
        return {
            'passed': passed,
            'error_count': len(issues),
            'details': '; '.join(issues) if issues else f'Temperature values valid in {len(temp_cols)} columns'
        }

    def _validate_efficiency_values(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Validate coulombic efficiency values"""
        efficiency_cols = [col for col in data.columns
                           if 'efficiency' in col.lower() or 'coulombic' in col.lower()]

        if not efficiency_cols:
            return {'passed': True, 'details': 'No efficiency columns found'}

        issues = []

        for col in efficiency_cols:
            col_data = data[col].filter(data[col].is_not_null())

            if col_data.is_empty():
                continue

            # Check for values outside reasonable range (0-1.5)
            below_zero = col_data.filter(col_data < 0)
            above_max = col_data.filter(col_data > 1.5)

            if not below_zero.is_empty():
                issues.append(f"{col}: {len(below_zero)} negative efficiency values")

            if not above_max.is_empty():
                issues.append(f"{col}: {len(above_max)} efficiency values > 1.5")

            # Check for exactly 1.0 values (might indicate calculation issues)
            exactly_one = col_data.filter(col_data == 1.0)
            if not exactly_one.is_empty() and len(exactly_one) > len(col_data) * 0.5:
                issues.append(
                    f"{col}: {len(exactly_one)} values exactly 1.0 ({len(exactly_one) / len(col_data) * 100:.1f}%)")

        passed = len(issues) == 0
        return {
            'passed': passed,
            'error_count': len(issues),
            'details': '; '.join(issues) if issues else f'Efficiency values valid in {len(efficiency_cols)} columns'
        }


# Convenience functions for common validation scenarios
def validate_battery_data(data: pl.DataFrame, strict: bool = False) -> DataQualityReport:
    """
    Validate battery data with standard rules.

    Args:
        data: DataFrame to validate
        strict: Whether to use strict validation (errors vs warnings)

    Returns:
        DataQualityReport
    """
    validator = DataValidator()
    validator.load_battery_standards()

    if strict:
        # Convert warnings to errors for strict validation
        for rule in validator.rules:
            if rule.severity == Severity.WARNING:
                rule.severity = Severity.ERROR

    return validator.validate(data)


def validate_cell_metadata(data: pl.DataFrame) -> DataQualityReport:
    """Validate cell metadata with specific rules"""
    validator = DataValidator()

    # Required fields for cell metadata
    validator.add_required_column('cell_id')
    validator.add_required_column('cell_name')

    # Cell ID should be unique and positive
    validator.add_custom_validation(
        'cell_id',
        lambda df: _validate_unique_ids(df, 'cell_id'),
        message="Cell IDs should be unique"
    )

    # Check for reasonable metadata values
    if 'total_active_mass_g' in data.columns:
        validator.add_range_check('total_active_mass_g', min_val=0.001, max_val=100.0)

    if 'nominal_capacity_ah' in data.columns:
        validator.add_range_check('nominal_capacity_ah', min_val=0.001, max_val=1000.0)

    return validator.validate(data)


def validate_cycle_data(data: pl.DataFrame) -> DataQualityReport:
    """Validate cycle data with specific rules"""
    validator = DataValidator()

    # Load standard battery rules
    validator.load_battery_standards()

    # Additional cycle-specific validations
    if 'regular_cycle_number' in data.columns:
        validator.add_custom_validation(
            'regular_cycle_number',
            lambda df: _validate_cycle_sequence(df),
            severity=Severity.WARNING,
            message="Cycle numbers should be sequential within each cell"
        )

    return validator.validate(data)


def quick_data_check(data: pl.DataFrame) -> Dict[str, Any]:
    """
    Perform a quick data quality check.

    Returns basic statistics and common issues.
    """
    if data.is_empty():
        return {
            'status': 'empty',
            'issues': ['DataFrame is empty'],
            'stats': {}
        }

    issues = []
    stats = {
        'rows': len(data),
        'columns': len(data.columns),
        'memory_mb': data.estimated_size() / (1024 * 1024),
        'null_counts': {},
        'dtypes': {}
    }

    # Check for common issues
    for col in data.columns:
        null_count = data[col].null_count()
        stats['null_counts'][col] = null_count
        stats['dtypes'][col] = str(data[col].dtype)

        # Flag columns with high null percentage
        null_pct = null_count / len(data) * 100
        if null_pct > 50:
            issues.append(f"Column '{col}' is {null_pct:.1f}% null")

        # Check for entirely null columns
        if null_count == len(data):
            issues.append(f"Column '{col}' is entirely null")

    # Check for duplicate rows
    unique_rows = data.unique().height
    if unique_rows != len(data):
        duplicate_count = len(data) - unique_rows
        issues.append(f"{duplicate_count} duplicate rows found")
        stats['duplicate_rows'] = duplicate_count

    # Check for required battery columns
    required_cols = ['cell_id']
    missing_required = [col for col in required_cols if col not in data.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")

    status = 'good' if len(issues) == 0 else 'issues_found'

    return {
        'status': status,
        'issues': issues,
        'stats': stats
    }


# Helper functions
def _validate_unique_ids(data: pl.DataFrame, column: str) -> Dict[str, Any]:
    """Check if IDs in a column are unique"""
    if column not in data.columns:
        return {'passed': False, 'details': f'Column {column} not found'}

    total_count = data[column].count()
    unique_count = data[column].unique().count()

    passed = total_count == unique_count
    duplicate_count = total_count - unique_count

    return {
        'passed': passed,
        'error_count': duplicate_count,
        'details': f'{duplicate_count} duplicate IDs found' if not passed else 'All IDs are unique'
    }


def _validate_cycle_sequence(data: pl.DataFrame) -> Dict[str, Any]:
    """Validate that cycle numbers are sequential within each cell"""
    if 'cell_id' not in data.columns or 'regular_cycle_number' not in data.columns:
        return {'passed': True, 'details': 'Required columns not found'}

    issues = []

    for cell_id in data['cell_id'].unique():
        cell_data = data.filter(pl.col('cell_id') == cell_id)
        cycles = cell_data['regular_cycle_number'].sort()

        if len(cycles) > 1:
            # Check for gaps in sequence
            expected_cycles = list(range(cycles.min(), cycles.max() + 1))
            actual_cycles = cycles.to_list()

            missing_cycles = set(expected_cycles) - set(actual_cycles)
            if missing_cycles:
                issues.append(f"Cell {cell_id}: missing cycles {sorted(missing_cycles)}")

    passed = len(issues) == 0
    return {
        'passed': passed,
        'error_count': len(issues),
        'details': '; '.join(issues) if issues else 'Cycle sequences look good'
    }


# Data quality metrics
def calculate_data_completeness(data: pl.DataFrame) -> Dict[str, float]:
    """Calculate completeness metrics for each column"""
    if data.is_empty():
        return {}

    completeness = {}
    total_rows = len(data)

    for col in data.columns:
        non_null_count = data[col].count()
        completeness[col] = non_null_count / total_rows

    return completeness


def calculate_data_consistency(data: pl.DataFrame) -> Dict[str, Any]:
    """Calculate consistency metrics"""
    metrics = {}

    if 'cell_id' in data.columns:
        # Check consistency of data per cell
        cell_counts = data.group_by('cell_id').count()
        metrics['cycles_per_cell'] = {
            'min': cell_counts['count'].min(),
            'max': cell_counts['count'].max(),
            'mean': cell_counts['count'].mean(),
            'std': cell_counts['count'].std()
        }

    # Check for data type consistency
    numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    metrics['numeric_columns'] = len(numeric_cols)
    metrics['total_columns'] = len(data.columns)
    metrics['numeric_ratio'] = len(numeric_cols) / len(data.columns)

    return metrics


# Export main classes and functions
__all__ = [
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
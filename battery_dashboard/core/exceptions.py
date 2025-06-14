# battery_dashboard/core/exceptions.py
class BatteryDashboardError(Exception):
    """Base exception for battery dashboard"""
    pass


class DataLoadingError(BatteryDashboardError):
    """Raised when data cannot be loaded"""
    pass


class DataValidationError(BatteryDashboardError):
    """Raised when data validation fails"""
    pass


class PlotConfigurationError(BatteryDashboardError):
    """Raised when plot configuration is invalid"""
    pass


class MLflowError(BatteryDashboardError):
    """Raised when MLflow operations fail"""
    pass


class CacheError(BatteryDashboardError):
    """Raised when cache operations fail"""
    pass

class DataProcessingError(BatteryDashboardError):
    """Raised when data processing fails"""
    pass


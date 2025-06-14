# battery_dashboard/core/plot_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
from pathlib import Path


class PlotType(Enum):
    LINE = "line"
    SCATTER = "scatter"
    AREA = "area"
    STEP = "step"
    BAR = "bar"


class AxisType(Enum):
    LINEAR = "linear"
    LOG = "log"


class ColorTheme(Enum):
    DEFAULT = "default"
    CATEGORY10 = "Category10"
    CATEGORY20 = "Category20"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"


class LegendPosition(Enum):
    RIGHT = "right"
    LEFT = "left"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class AxisConfig:
    """Configuration for plot axes"""
    label: Optional[str] = None
    type: AxisType = AxisType.LINEAR
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'type': self.type.value,
            'min': self.min_value,
            'max': self.max_value
        }


@dataclass
class SeriesConfig:
    """Configuration for individual data series"""
    visible: bool = True
    color: Optional[str] = None
    line_width: int = 2
    line_style: str = "solid"  # solid, dashed, dotted, dotdash
    marker_size: int = 6
    marker_shape: str = "circle"  # circle, square, triangle, diamond, cross, x, star
    opacity: float = 0.8
    y_axis: str = "primary"  # primary, secondary

    def to_dict(self) -> Dict[str, Any]:
        return {
            'visible': self.visible,
            'color': self.color,
            'line_width': self.line_width,
            'line_style': self.line_style,
            'marker_size': self.marker_size,
            'marker_shape': self.marker_shape,
            'opacity': self.opacity,
            'y_axis': self.y_axis
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeriesConfig':
        return cls(**data)


@dataclass
class PlotConfig:
    """Comprehensive plot configuration"""
    # Basic plot settings
    plot_type: PlotType = PlotType.LINE
    x_axis_column: str = "regular_cycle_number"
    y_axis_column: str = "discharge_capacity"
    y_axis_secondary_column: Optional[str] = None
    group_by_column: str = "cell_id"

    # Appearance
    plot_height: int = 600
    plot_width: int = 800
    color_theme: ColorTheme = ColorTheme.DEFAULT
    title: Optional[str] = None

    # Grid and legend
    show_grid: bool = True
    grid_color: str = "#e0e0e0"
    show_legend: bool = True
    legend_position: LegendPosition = LegendPosition.RIGHT

    # Axes configuration
    x_axis: AxisConfig = field(default_factory=AxisConfig)
    y_axis: AxisConfig = field(default_factory=AxisConfig)
    y_axis_secondary: AxisConfig = field(default_factory=AxisConfig)

    # Series configurations (keyed by group name)
    series_configs: Dict[str, SeriesConfig] = field(default_factory=dict)

    # Advanced options
    use_datashader: bool = True
    background_color: str = "#ffffff"
    tools: List[str] = field(default_factory=lambda: [
        "pan", "wheel_zoom", "box_zoom", "reset", "hover"
    ])

    def to_hvplot_kwargs(self, data_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert configuration to hvplot parameters"""
        kwargs = {
            'kind': self.plot_type.value,
            'x': self.x_axis_column,
            'y': self.y_axis_column,
            'by': self.group_by_column,
            'height': self.plot_height,
            'width': self.plot_width,
            'grid': self.show_grid,
            'legend': self.show_legend,
            'tools': self.tools,
            'logx': self.x_axis.type == AxisType.LOG,
            'logy': self.y_axis.type == AxisType.LOG,
        }

        # Add color theme
        if self.color_theme != ColorTheme.DEFAULT:
            kwargs['cmap'] = self.color_theme.value

        # Add axis labels
        kwargs['xlabel'] = self.x_axis.label or self.x_axis_column.replace('_', ' ').title()
        kwargs['ylabel'] = self.y_axis.label or self.y_axis_column.replace('_', ' ').title()

        # Add title
        if self.title:
            kwargs['title'] = self.title

        # Add axis limits
        if self.x_axis.min_value is not None and self.x_axis.max_value is not None:
            kwargs['xlim'] = (self.x_axis.min_value, self.x_axis.max_value)

        if self.y_axis.min_value is not None and self.y_axis.max_value is not None:
            kwargs['ylim'] = (self.y_axis.min_value, self.y_axis.max_value)

        return kwargs

    def get_secondary_axis_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for secondary y-axis plot"""
        kwargs = self.to_hvplot_kwargs()
        kwargs.update({
            'y': self.y_axis_secondary_column,
            'ylabel': (self.y_axis_secondary.label or
                       self.y_axis_secondary_column.replace('_', ' ').title()),
            'yaxis': 'right',
            'legend': False,
            'logy': self.y_axis_secondary.type == AxisType.LOG,
        })

        if (self.y_axis_secondary.min_value is not None and
                self.y_axis_secondary.max_value is not None):
            kwargs['ylim'] = (
                self.y_axis_secondary.min_value,
                self.y_axis_secondary.max_value
            )

        return kwargs

    def validate(self, available_columns: Optional[List[str]] = None) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate dimensions
        if self.plot_height < 300:
            errors.append("Plot height must be at least 300px")
        if self.plot_width < 400:
            errors.append("Plot width must be at least 400px")

        # Validate columns if available_columns provided
        if available_columns:
            required_columns = [
                self.x_axis_column,
                self.y_axis_column,
                self.group_by_column
            ]
            if self.y_axis_secondary_column:
                required_columns.append(self.y_axis_secondary_column)

            missing_columns = [col for col in required_columns
                               if col not in available_columns]
            if missing_columns:
                errors.append(f"Missing columns: {', '.join(missing_columns)}")

        # Validate axis ranges
        if (self.x_axis.min_value is not None and
                self.x_axis.max_value is not None and
                self.x_axis.min_value >= self.x_axis.max_value):
            errors.append("X-axis minimum must be less than maximum")

        if (self.y_axis.min_value is not None and
                self.y_axis.max_value is not None and
                self.y_axis.min_value >= self.y_axis.max_value):
            errors.append("Y-axis minimum must be less than maximum")

        return errors

    def update_series_config(self, group_name: str, config: SeriesConfig):
        """Update configuration for a specific series"""
        self.series_configs[str(group_name)] = config

    def get_series_config(self, group_name: str) -> SeriesConfig:
        """Get configuration for a specific series"""
        return self.series_configs.get(str(group_name), SeriesConfig())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary"""
        return {
            'plot_type': self.plot_type.value,
            'x_axis_column': self.x_axis_column,
            'y_axis_column': self.y_axis_column,
            'y_axis_secondary_column': self.y_axis_secondary_column,
            'group_by_column': self.group_by_column,
            'plot_height': self.plot_height,
            'plot_width': self.plot_width,
            'color_theme': self.color_theme.value,
            'title': self.title,
            'show_grid': self.show_grid,
            'grid_color': self.grid_color,
            'show_legend': self.show_legend,
            'legend_position': self.legend_position.value,
            'x_axis': self.x_axis.to_dict(),
            'y_axis': self.y_axis.to_dict(),
            'y_axis_secondary': self.y_axis_secondary.to_dict(),
            'series_configs': {k: v.to_dict() for k, v in self.series_configs.items()},
            'use_datashader': self.use_datashader,
            'background_color': self.background_color,
            'tools': self.tools
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotConfig':
        """Deserialize configuration from dictionary"""
        # Convert enum strings back to enums
        data['plot_type'] = PlotType(data['plot_type'])
        data['color_theme'] = ColorTheme(data['color_theme'])
        data['legend_position'] = LegendPosition(data['legend_position'])

        # Convert axis configs
        data['x_axis'] = AxisConfig(
            label=data['x_axis']['label'],
            type=AxisType(data['x_axis']['type']),
            min_value=data['x_axis']['min'],
            max_value=data['x_axis']['max']
        )
        data['y_axis'] = AxisConfig(
            label=data['y_axis']['label'],
            type=AxisType(data['y_axis']['type']),
            min_value=data['y_axis']['min'],
            max_value=data['y_axis']['max']
        )
        data['y_axis_secondary'] = AxisConfig(
            label=data['y_axis_secondary']['label'],
            type=AxisType(data['y_axis_secondary']['type']),
            min_value=data['y_axis_secondary']['min'],
            max_value=data['y_axis_secondary']['max']
        )

        # Convert series configs
        data['series_configs'] = {
            k: SeriesConfig.from_dict(v)
            for k, v in data['series_configs'].items()
        }

        return cls(**data)

    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'PlotConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> 'PlotConfig':
        """Create a deep copy of the configuration"""
        return self.from_dict(self.to_dict())


# Preset configurations for common use cases
PRESET_CONFIGS = {
    'capacity_retention': PlotConfig(
        x_axis_column='regular_cycle_number',
        y_axis_column='discharge_capacity',
        title='Discharge Capacity vs Cycle Number',
        x_axis=AxisConfig(label='Cycle Number'),
        y_axis=AxisConfig(label='Discharge Capacity (Ah)')
    ),

    'efficiency': PlotConfig(
        x_axis_column='regular_cycle_number',
        y_axis_column='coulombic_efficiency',
        title='Coulombic Efficiency vs Cycle Number',
        x_axis=AxisConfig(label='Cycle Number'),
        y_axis=AxisConfig(label='Coulombic Efficiency (%)', min_value=95, max_value=100)
    ),

    'specific_capacity': PlotConfig(
        x_axis_column='regular_cycle_number',
        y_axis_column='discharge_capacity_specific_mAh_g',
        title='Specific Discharge Capacity vs Cycle Number',
        x_axis=AxisConfig(label='Cycle Number'),
        y_axis=AxisConfig(label='Specific Capacity (mAh/g)')
    ),

    'dual_axis_capacity_efficiency': PlotConfig(
        x_axis_column='regular_cycle_number',
        y_axis_column='discharge_capacity',
        y_axis_secondary_column='coulombic_efficiency',
        title='Capacity and Efficiency vs Cycle Number',
        x_axis=AxisConfig(label='Cycle Number'),
        y_axis=AxisConfig(label='Discharge Capacity (Ah)'),
        y_axis_secondary=AxisConfig(label='Coulombic Efficiency (%)')
    )
}
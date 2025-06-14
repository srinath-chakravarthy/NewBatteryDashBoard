# battery_dashboard/ui/components/cycle_plots.py
import panel as pn
import polars as pl
import holoviews as hv
import hvplot.polars
from typing import List, Optional, Dict, Any
import pandas as pd

from ..base_tab import BaseTab
from ...core.plot_config import PlotConfig, PlotType, ColorTheme, AxisConfig, SeriesConfig, PRESET_CONFIGS
from ...core.settings import SettingsPanel, SettingSpec, WidgetType, PLOT_BASIC_SETTINGS, PLOT_AXIS_SETTINGS, \
    PLOT_ADVANCED_SETTINGS
from ...core.state_manager import AppStateManager
from ...utils.logging import get_logger

# Configure HoloViews
hv.extension('bokeh')
logger = get_logger(__name__)


class CyclePlotsTab(BaseTab):
    """Enhanced cycle plots tab using PlotConfig and SettingsPanel"""

    def __init__(self, state_manager: AppStateManager, **params):
        # Initialize with default plot configuration
        self.plot_config = PRESET_CONFIGS['capacity_retention'].copy()
        self.current_plot = None

        super().__init__(state_manager, title="Cycle Plots", **params)

    def setup_tab(self):
        """Setup tab-specific components"""
        self.create_plot_controls()
        self.create_settings_panels()

        # Initialize with empty state
        self.show_empty_state("No cells selected. Please select cells in the Cell Selector tab.")

    def create_plot_controls(self):
        """Create basic plot control widgets"""
        # Get available columns (will be updated when data loads)
        available_columns = ["regular_cycle_number", "discharge_capacity", "charge_capacity",
                             "coulombic_efficiency", "energy_efficiency"]

        # Basic controls
        self.plot_type_select = pn.widgets.Select(
            name="Plot Type",
            options=[pt.value for pt in PlotType],
            value=self.plot_config.plot_type.value,
            width=200
        )

        self.x_axis_select = pn.widgets.Select(
            name="X-Axis",
            options=available_columns,
            value=self.plot_config.x_axis_column,
            width=200
        )

        self.y_axis_select = pn.widgets.Select(
            name="Y-Axis",
            options=available_columns,
            value=self.plot_config.y_axis_column,
            width=200
        )

        self.y_axis_secondary_select = pn.widgets.Select(
            name="Secondary Y-Axis",
            options=["None"] + available_columns,
            value="None",
            width=200
        )

        self.group_by_select = pn.widgets.Select(
            name="Group By",
            options=["cell_id", "cell_name"],
            value=self.plot_config.group_by_column,
            width=200
        )

        # Preset buttons
        self.preset_buttons = pn.Row(
            *[pn.widgets.Button(name=name.replace('_', ' ').title(), width=120)
              for name in PRESET_CONFIGS.keys()]
        )

        # Export buttons
        self.export_png_btn = pn.widgets.Button(name="Export PNG", button_type="success", width=120)
        self.export_svg_btn = pn.widgets.Button(name="Export SVG", button_type="success", width=120)

        # Setup event handlers
        self._setup_control_handlers()

    def create_settings_panels(self):
        """Create advanced settings panels using SettingsPanel"""
        # Combine all settings specs
        all_settings = PLOT_BASIC_SETTINGS + PLOT_AXIS_SETTINGS + PLOT_ADVANCED_SETTINGS

        # Create settings panel
        self.settings_panel = SettingsPanel(all_settings, title="Plot Settings")

        # Create series settings (will be populated when data is available)
        self.series_settings_panel = None

        # Create cards
        self.advanced_settings_card = self.settings_panel.create_card(
            collapsed=True,
            collapsible=True
        )

        self.series_settings_card = pn.Card(
            pn.pane.Markdown("*Load data to configure series settings*"),
            title="Series Settings",
            collapsed=True,
            collapsible=True,
            header_background="#3498db",
            header_color="white"
        )

    def _setup_control_handlers(self):
        """Setup event handlers for controls"""
        self.plot_type_select.param.watch(self._on_plot_type_change, 'value')
        self.x_axis_select.param.watch(self._on_axis_change, 'value')
        self.y_axis_select.param.watch(self._on_axis_change, 'value')
        self.y_axis_secondary_select.param.watch(self._on_axis_change, 'value')
        self.group_by_select.param.watch(self._on_group_by_change, 'value')

        # Preset button handlers
        for i, (preset_name, preset_config) in enumerate(PRESET_CONFIGS.items()):
            self.preset_buttons[i].on_click(lambda event, name=preset_name: self._apply_preset(name))

        # Export handlers
        self.export_png_btn.on_click(self._export_png)
        self.export_svg_btn.on_click(self._export_svg)

        # Settings panel handlers
        for widget in self.settings_panel.widgets.values():
            widget.param.watch(self._on_settings_change, 'value')

    def create_controls(self) -> pn.Column:
        """Create sidebar controls"""
        return pn.Column(
            pn.pane.Markdown("## Plot Configuration"),
            self.plot_type_select,
            self.x_axis_select,
            self.y_axis_select,
            self.y_axis_secondary_select,
            pn.layout.Divider(),
            pn.pane.Markdown("## Grouping"),
            self.group_by_select,
            pn.layout.Divider(),
            pn.pane.Markdown("## Presets"),
            self.preset_buttons,
            pn.layout.Divider(),
            pn.pane.Markdown("## Export"),
            pn.Row(self.export_png_btn, self.export_svg_btn),
            width=280
        )

    def create_main_content(self) -> pn.layout.Panel:
        """Create main content area"""
        return pn.Column(
            pn.pane.Markdown("# Cycle Analysis", styles={"margin-bottom": "10px"}),
            self.container,
            self.advanced_settings_card,
            self.series_settings_card,
            sizing_mode="stretch_both"
        )

    def on_cycle_data_change(self, cycle_data: Optional[pl.DataFrame]):
        """Handle cycle data changes"""
        if cycle_data is None or cycle_data.is_empty():
            self.show_empty_state("No cycle data available.")
            return

        logger.info(f"Received cycle data: {len(cycle_data)} rows")

        # Update available columns
        self._update_column_options(cycle_data)

        # Create series settings panel
        self._create_series_settings_panel(cycle_data)

        # Generate initial plot
        self._update_plot()

    def _update_column_options(self, cycle_data: pl.DataFrame):
        """Update dropdown options based on available data"""
        # Get numeric columns for axes
        numeric_cols = [
            col for col in cycle_data.columns
            if cycle_data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

        # Get categorical columns for grouping
        categorical_cols = [
            col for col in cycle_data.columns
            if cycle_data[col].dtype in [pl.Utf8, pl.String] or col in ['cell_id', 'cell_name']
        ]

        # Update widget options
        self.x_axis_select.options = sorted(numeric_cols)
        self.y_axis_select.options = sorted(numeric_cols)
        self.y_axis_secondary_select.options = ["None"] + sorted(numeric_cols)
        self.group_by_select.options = sorted(categorical_cols)

        # Update plot config if current selections are invalid
        if self.plot_config.x_axis_column not in numeric_cols:
            self.plot_config.x_axis_column = numeric_cols[0] if numeric_cols else "regular_cycle_number"
            self.x_axis_select.value = self.plot_config.x_axis_column

        if self.plot_config.y_axis_column not in numeric_cols:
            capacity_cols = [col for col in numeric_cols if 'discharge_capacity' in col]
            self.plot_config.y_axis_column = capacity_cols[0] if capacity_cols else numeric_cols[0]
            self.y_axis_select.value = self.plot_config.y_axis_column

        if self.plot_config.group_by_column not in categorical_cols:
            self.plot_config.group_by_column = 'cell_id' if 'cell_id' in categorical_cols else categorical_cols[0]
            self.group_by_select.value = self.plot_config.group_by_column

    def _create_series_settings_panel(self, cycle_data: pl.DataFrame):
        """Create series settings panel based on current grouping"""
        # Get unique groups
        unique_groups = cycle_data[self.plot_config.group_by_column].unique().to_list()

        if len(unique_groups) > 20:
            # Too many series - show a warning instead
            self.series_settings_card.objects = [
                pn.pane.Alert(
                    f"Too many series ({len(unique_groups)}) for individual configuration. "
                    "Consider changing the grouping column.",
                    alert_type="warning"
                )
            ]
            return

        # Create series settings for each group
        series_data = {
            "group": [],
            "visible": [],
            "color": [],
            "line_width": [],
            "line_style": [],
            "marker_size": [],
            "marker_shape": [],
            "opacity": []
        }

        for group in unique_groups:
            group_str = str(group)
            config = self.plot_config.get_series_config(group_str)

            series_data["group"].append(group_str)
            series_data["visible"].append(config.visible)
            series_data["color"].append(config.color or "")
            series_data["line_width"].append(config.line_width)
            series_data["line_style"].append(config.line_style)
            series_data["marker_size"].append(config.marker_size)
            series_data["marker_shape"].append(config.marker_shape)
            series_data["opacity"].append(config.opacity)

        # Create tabulator for series settings
        series_df = pd.DataFrame(series_data)

        self.series_table = pn.widgets.Tabulator(
            series_df,
            height=400,
            layout="fit_data_fill",
            editors={
                "visible": {"type": "tickCross"},
                "color": {"type": "color"},
                "line_width": {"type": "number", "min": 1, "max": 10, "step": 1},
                "line_style": {"type": "list", "values": ["solid", "dashed", "dotted", "dotdash"]},
                "marker_size": {"type": "number", "min": 1, "max": 20, "step": 1},
                "marker_shape": {"type": "list",
                                 "values": ["circle", "square", "triangle", "diamond", "cross", "x", "star"]},
                "opacity": {"type": "number", "min": 0.1, "max": 1.0, "step": 0.1}
            },
            show_index=False
        )

        # Create apply/reset buttons
        apply_btn = pn.widgets.Button(name="Apply Changes", button_type="success")
        reset_btn = pn.widgets.Button(name="Reset to Default", button_type="default")

        apply_btn.on_click(self._apply_series_settings)
        reset_btn.on_click(self._reset_series_settings)

        # Update series settings card
        self.series_settings_card.objects = [
            pn.Column(
                pn.pane.Markdown("Configure individual series appearance:"),
                self.series_table,
                pn.Row(apply_btn, reset_btn, align="end")
            )
        ]

    def _on_plot_type_change(self, event):
        """Handle plot type change"""
        self.plot_config.plot_type = PlotType(event.new)
        self._update_plot()

    def _on_axis_change(self, event):
        """Handle axis selection change"""
        if event.obj == self.x_axis_select:
            self.plot_config.x_axis_column = event.new
        elif event.obj == self.y_axis_select:
            self.plot_config.y_axis_column = event.new
        elif event.obj == self.y_axis_secondary_select:
            self.plot_config.y_axis_secondary_column = event.new if event.new != "None" else None

        self._update_plot()

    def _on_group_by_change(self, event):
        """Handle group by change"""
        self.plot_config.group_by_column = event.new

        # Recreate series settings panel
        cycle_data = self.get_cycle_data()
        if cycle_data is not None:
            self._create_series_settings_panel(cycle_data)

        self._update_plot()

    def _on_settings_change(self, event):
        """Handle advanced settings change"""
        settings_values = self.settings_panel.get_values()

        # Update plot config from settings
        self.plot_config.plot_height = settings_values.get('plot_height', 600)
        self.plot_config.plot_width = settings_values.get('plot_width', 800)
        self.plot_config.show_grid = settings_values.get('show_grid', True)
        self.plot_config.show_legend = settings_values.get('show_legend', True)
        self.plot_config.title = settings_values.get('plot_title', '')

        # Update axis configs
        self.plot_config.x_axis.label = settings_values.get('x_axis_label', '')
        self.plot_config.x_axis.type = settings_values.get('x_axis_type', 'linear')
        self.plot_config.x_axis.min_value = settings_values.get('x_axis_min')
        self.plot_config.x_axis.max_value = settings_values.get('x_axis_max')

        self.plot_config.y_axis.label = settings_values.get('y_axis_label', '')
        self.plot_config.y_axis.type = settings_values.get('y_axis_type', 'linear')
        self.plot_config.y_axis.min_value = settings_values.get('y_axis_min')
        self.plot_config.y_axis.max_value = settings_values.get('y_axis_max')

        self._update_plot()

    def _apply_preset(self, preset_name: str):
        """Apply a preset configuration"""
        if preset_name in PRESET_CONFIGS:
            self.plot_config = PRESET_CONFIGS[preset_name].copy()

            # Update UI controls
            self.plot_type_select.value = self.plot_config.plot_type.value
            self.x_axis_select.value = self.plot_config.x_axis_column
            self.y_axis_select.value = self.plot_config.y_axis_column
            self.y_axis_secondary_select.value = self.plot_config.y_axis_secondary_column or "None"
            self.group_by_select.value = self.plot_config.group_by_column

            # Update settings panel
            settings_values = {
                'plot_title': self.plot_config.title or '',
                'plot_height': self.plot_config.plot_height,
                'plot_width': self.plot_config.plot_width,
                'show_grid': self.plot_config.show_grid,
                'show_legend': self.plot_config.show_legend,
                'x_axis_label': self.plot_config.x_axis.label or '',
                'y_axis_label': self.plot_config.y_axis.label or '',
            }
            self.settings_panel.set_values(settings_values)

            self._update_plot()

    def _apply_series_settings(self, event):
        """Apply series settings from table"""
        if not hasattr(self, 'series_table'):
            return

        table_data = self.series_table.value

        for _, row in table_data.iterrows():
            group_str = str(row['group'])
            config = SeriesConfig(
                visible=row['visible'],
                color=row['color'] if row['color'] else None,
                line_width=row['line_width'],
                line_style=row['line_style'],
                marker_size=row['marker_size'],
                marker_shape=row['marker_shape'],
                opacity=row['opacity']
            )
            self.plot_config.update_series_config(group_str, config)

        self._update_plot()

    def _reset_series_settings(self, event):
        """Reset series settings to defaults"""
        self.plot_config.series_configs = {}

        # Recreate series settings panel
        cycle_data = self.get_cycle_data()
        if cycle_data is not None:
            self._create_series_settings_panel(cycle_data)

        self._update_plot()

    def _update_plot(self):
        """Update the plot based on current configuration"""
        cycle_data = self.get_cycle_data()
        if cycle_data is None or cycle_data.is_empty():
            self.show_empty_state("No cycle data available for plotting.")
            return

        try:
            # Validate configuration
            errors = self.plot_config.validate(cycle_data.columns)
            if errors:
                self.show_error(f"Configuration errors: {', '.join(errors)}")
                return

            # Create plot using hvplot
            plot_kwargs = self.plot_config.to_hvplot_kwargs()

            # Create main plot
            main_plot = cycle_data.hvplot(**plot_kwargs)

            # Add secondary axis if configured
            if self.plot_config.y_axis_secondary_column:
                secondary_kwargs = self.plot_config.get_secondary_axis_kwargs()
                secondary_plot = cycle_data.hvplot(**secondary_kwargs)

                # Combine plots
                combined_plot = (main_plot * secondary_plot).opts(multi_y=True)
                self.current_plot = combined_plot
            else:
                self.current_plot = main_plot

            # Update container
            self.container.objects = [self.current_plot]

            # Save configuration to state
            self.state_manager.save_plot_config("cycle_plots", self.plot_config.to_dict())

        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            self.show_error(f"Error creating plot: {str(e)}")

    def _export_png(self, event):
        """Export current plot as PNG"""
        if self.current_plot is None:
            return

        try:
            # Implementation depends on your export requirements
            # This is a placeholder
            self.container.append(
                pn.pane.Alert("PNG export functionality to be implemented", alert_type="info")
            )
        except Exception as e:
            logger.error(f"Error exporting PNG: {e}")

    def _export_svg(self, event):
        """Export current plot as SVG"""
        if self.current_plot is None:
            return

        try:
            # Implementation depends on your export requirements
            # This is a placeholder
            self.container.append(
                pn.pane.Alert("SVG export functionality to be implemented", alert_type="info")
            )
        except Exception as e:
            logger.error(f"Error exporting SVG: {e}")
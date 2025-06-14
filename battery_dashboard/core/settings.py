# battery_dashboard/core/settings.py
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Union, Callable, Tuple
from enum import Enum
import panel as pn
import param


class WidgetType(Enum):
    TEXT_INPUT = "text_input"
    NUMBER_INPUT = "number_input"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    CHECKBOX = "checkbox"
    INT_SLIDER = "int_slider"
    FLOAT_SLIDER = "float_slider"
    COLOR_PICKER = "color_picker"
    DATE_PICKER = "date_picker"
    MULTI_CHOICE = "multi_choice"


@dataclass
class SettingSpec:
    """Specification for a single setting widget"""
    name: str
    widget_type: WidgetType
    default: Any
    label: Optional[str] = None
    description: str = ""
    options: Optional[List[Any]] = None
    bounds: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    step: Optional[Union[int, float]] = None
    width: int = 200
    validator: Optional[Callable[[Any], Union[bool, str]]] = None
    depends_on: Optional[str] = None  # Enable/disable based on another setting
    section: str = "General"  # Group settings into sections
    tooltip = None

    def __post_init__(self):
        if self.label is None:
            # Convert snake_case to Title Case
            self.label = self.name.replace('_', ' ').title()


class SettingsPanel:
    """Dynamic settings panel generator from specifications"""

    def __init__(self, specs: List[SettingSpec], title: str = "Settings"):
        self.specs = specs
        self.title = title
        self.widgets: Dict[str, pn.param.Parameter] = {}
        self.sections: Dict[str, List[SettingSpec]] = {}
        self.validation_errors: Dict[str, str] = {}

        # Group specs by section
        for spec in specs:
            if spec.section not in self.sections:
                self.sections[spec.section] = []
            self.sections[spec.section].append(spec)

        # Create widgets
        self._create_widgets()
        self._setup_dependencies()

    def _create_widgets(self):
        """Create Panel widgets from specifications"""
        for spec in self.specs:
            widget = self._create_widget(spec)
            self.widgets[spec.name] = widget

            # Add validation if specified
            if spec.validator:
                widget.param.watch(
                    lambda event, s=spec: self._validate_setting(s, event.new),
                    'value'
                )

    def _create_widget(self, spec):
        """Create a widget for the given setting spec"""
        # Base kwargs from spec
        base_kwargs = {
            "name": spec.label,
            "value": spec.default,
            "disabled": not getattr(spec, "enabled", True)
        }

        # Add tooltip if provided
        if hasattr(spec, 'tooltip') and spec.tooltip:
            base_kwargs["tooltips"] = spec.tooltip

        # Additional custom arguments
        if spec.base_kwargs:
            # Handle 'description' for Checkbox - convert to 'name'
            if spec.widget_type == "checkbox" and "description" in spec.widget_kwargs:
                base_kwargs["name"] = spec.widget_kwargs.pop("description")

            base_kwargs.update(spec.widget_kwargs)

        # Create appropriate widget based on type
        if spec.widget_type == "checkbox":
            return pn.widgets.Checkbox(**base_kwargs)
        elif spec.widget_type == "select":
            return pn.widgets.Select(options=spec.options, **base_kwargs)
        elif spec.widget_type == "multiselect":
            return pn.widgets.MultiSelect(options=spec.options, **base_kwargs)
        elif spec.widget_type == "slider":
            return pn.widgets.FloatSlider(start=spec.min_value, end=spec.max_value,
                                          step=spec.step, **base_kwargs)
        elif spec.widget_type == "range_slider":
            return pn.widgets.RangeSlider(start=spec.min_value, end=spec.max_value,
                                          step=spec.step, **base_kwargs)
        elif spec.widget_type == "numeric":
            return pn.widgets.FloatInput(**base_kwargs)
        elif spec.widget_type == "text":
            return pn.widgets.TextInput(**base_kwargs)
        elif spec.widget_type == "color":
            return pn.widgets.ColorPicker(**base_kwargs)
        else:
            # Default to text input for unknown types
            return pn.widgets.TextInput(**base_kwargs)

    def _setup_dependencies(self):
        """Setup widget dependencies (enable/disable based on other widgets)"""
        for spec in self.specs:
            if spec.depends_on and spec.depends_on in self.widgets:
                # Watch the dependency widget and update this widget accordingly
                dependency_widget = self.widgets[spec.depends_on]
                target_widget = self.widgets[spec.name]

                def update_widget_state(event, target=target_widget):
                    target.disabled = not bool(event.new)

                dependency_widget.param.watch(update_widget_state, 'value')

                # Set initial state
                target_widget.disabled = not bool(dependency_widget.value)

    def _validate_setting(self, spec: SettingSpec, value: Any) -> bool:
        """Validate a setting value"""
        if spec.validator:
            result = spec.validator(value)
            if isinstance(result, str):
                # Validation failed with error message
                self.validation_errors[spec.name] = result
                return False
            elif not result:
                # Validation failed without specific message
                self.validation_errors[spec.name] = f"Invalid value for {spec.label}"
                return False
            else:
                # Validation passed
                if spec.name in self.validation_errors:
                    del self.validation_errors[spec.name]
                return True
        return True

    def get_values(self) -> Dict[str, Any]:
        """Get current values from all widgets"""
        return {name: widget.value for name, widget in self.widgets.items()}

    def set_values(self, values: Dict[str, Any]):
        """Set values for multiple widgets"""
        for name, value in values.items():
            if name in self.widgets:
                self.widgets[name].value = value

    def get_widget(self, name: str) -> Optional[pn.widgets.Widget]:
        """Get a specific widget by name"""
        return self.widgets.get(name)

    def validate_all(self) -> bool:
        """Validate all settings and return True if all are valid"""
        all_valid = True
        for spec in self.specs:
            if spec.name in self.widgets:
                current_value = self.widgets[spec.name].value
                if not self._validate_setting(spec, current_value):
                    all_valid = False
        return all_valid

    def get_validation_errors(self) -> Dict[str, str]:
        """Get all current validation errors"""
        return self.validation_errors.copy()

    def create_layout(self, as_tabs: bool = True) -> pn.layout.Panel:
        """Create the Panel layout for the settings"""
        if as_tabs and len(self.sections) > 1:
            # Create tabs for different sections
            tabs = []
            for section_name, section_specs in self.sections.items():
                section_widgets = []
                for spec in section_specs:
                    widget = self.widgets[spec.name]
                    if spec.description:
                        section_widgets.append(
                            pn.Column(
                                widget,
                                pn.pane.Markdown(
                                    f"*{spec.description}*",
                                    styles={'font-size': '0.9em', 'color': '#666'}
                                ),
                                margin=(0, 0, 10, 0)
                            )
                        )
                    else:
                        section_widgets.append(widget)

                tabs.append((section_name, pn.Column(*section_widgets)))

            return pn.Tabs(*tabs)

        else:
            # Single column layout
            all_widgets = []
            current_section = None

            for spec in self.specs:
                # Add section header if new section
                if spec.section != current_section:
                    current_section = spec.section
                    if len(self.sections) > 1:  # Only show headers if multiple sections
                        all_widgets.append(
                            pn.pane.Markdown(f"## {current_section}")
                        )

                widget = self.widgets[spec.name]
                if spec.description:
                    all_widgets.append(
                        pn.Column(
                            widget,
                            pn.pane.Markdown(
                                f"*{spec.description}*",
                                styles={'font-size': '0.9em', 'color': '#666'}
                            ),
                            margin=(0, 0, 10, 0)
                        )
                    )
                else:
                    all_widgets.append(widget)

            return pn.Column(*all_widgets)

    def create_card(self, collapsed: bool = True, collapsible: bool = True) -> pn.Card:
        """Create a Card containing the settings"""
        layout = self.create_layout()

        return pn.Card(
            layout,
            title=self.title,
            collapsed=collapsed,
            collapsible=collapsible,
            header_background="#3498db",
            header_color="white"
        )


# Common validation functions
def validate_positive_number(value: Union[int, float]) -> Union[bool, str]:
    """Validate that a number is positive"""
    if value is None:
        return True  # Allow None values
    if isinstance(value, (int, float)) and value > 0:
        return True
    return "Value must be positive"


def validate_percentage(value: Union[int, float]) -> Union[bool, str]:
    """Validate that a value is between 0 and 100"""
    if value is None:
        return True
    if isinstance(value, (int, float)) and 0 <= value <= 100:
        return True
    return "Value must be between 0 and 100"


def validate_range(min_val: Union[int, float], max_val: Union[int, float]):
    """Create a validator for a numeric range"""

    def validator(value: Union[int, float]) -> Union[bool, str]:
        if value is None:
            return True
        if isinstance(value, (int, float)) and min_val <= value <= max_val:
            return True
        return f"Value must be between {min_val} and {max_val}"

    return validator


# Predefined setting specifications for common use cases
PLOT_BASIC_SETTINGS = [
    SettingSpec(
        name="plot_title",
        widget_type=WidgetType.TEXT_INPUT,
        default="",
        label="Plot Title",
        description="Main title for the plot",
        section="Basic"
    ),
    SettingSpec(
        name="plot_height",
        widget_type=WidgetType.INT_SLIDER,
        default=600,
        bounds=(300, 1200),
        step=50,
        label="Plot Height (px)",
        validator=validate_positive_number,
        section="Basic"
    ),
    SettingSpec(
        name="plot_width",
        widget_type=WidgetType.INT_SLIDER,
        default=800,
        bounds=(400, 2000),
        step=50,
        label="Plot Width (px)",
        validator=validate_positive_number,
        section="Basic"
    ),
    SettingSpec(
        name="show_grid",
        widget_type=WidgetType.CHECKBOX,
        default=True,
        label="Show Grid",
        section="Basic"
    ),
    SettingSpec(
        name="show_legend",
        widget_type=WidgetType.CHECKBOX,
        default=True,
        label="Show Legend",
        section="Basic"
    )
]

PLOT_AXIS_SETTINGS = [
    SettingSpec(
        name="x_axis_label",
        widget_type=WidgetType.TEXT_INPUT,
        default="",
        label="X-Axis Label",
        section="X-Axis"
    ),
    SettingSpec(
        name="x_axis_type",
        widget_type=WidgetType.SELECT,
        default="linear",
        options=["linear", "log"],
        label="X-Axis Type",
        section="X-Axis"
    ),
    SettingSpec(
        name="x_axis_min",
        widget_type=WidgetType.NUMBER_INPUT,
        default=None,
        label="X-Axis Minimum",
        step=0.1,
        section="X-Axis"
    ),
    SettingSpec(
        name="x_axis_max",
        widget_type=WidgetType.NUMBER_INPUT,
        default=None,
        label="X-Axis Maximum",
        step=0.1,
        section="X-Axis"
    ),
    SettingSpec(
        name="y_axis_label",
        widget_type=WidgetType.TEXT_INPUT,
        default="",
        label="Y-Axis Label",
        section="Y-Axis"
    ),
    SettingSpec(
        name="y_axis_type",
        widget_type=WidgetType.SELECT,
        default="linear",
        options=["linear", "log"],
        label="Y-Axis Type",
        section="Y-Axis"
    ),
    SettingSpec(
        name="y_axis_min",
        widget_type=WidgetType.NUMBER_INPUT,
        default=None,
        label="Y-Axis Minimum",
        step=0.1,
        section="Y-Axis"
    ),
    SettingSpec(
        name="y_axis_max",
        widget_type=WidgetType.NUMBER_INPUT,
        default=None,
        label="Y-Axis Maximum",
        step=0.1,
        section="Y-Axis"
    )
]

PLOT_ADVANCED_SETTINGS = [
    SettingSpec(
        name="grid_color",
        widget_type=WidgetType.COLOR_PICKER,
        default="#e0e0e0",
        label="Grid Color",
        depends_on="show_grid",
        section="Advanced"
    ),
    SettingSpec(
        name="background_color",
        widget_type=WidgetType.COLOR_PICKER,
        default="#ffffff",
        label="Background Color",
        section="Advanced"
    ),
    SettingSpec(
        name="legend_position",
        widget_type=WidgetType.SELECT,
        default="right",
        options=["right", "left", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"],
        label="Legend Position",
        depends_on="show_legend",
        section="Advanced"
    ),
    SettingSpec(
        name="use_datashader",
        widget_type=WidgetType.CHECKBOX,
        default=True,
        label="Use Datashader",
        description="Use datashader for large datasets (>1000 points)",
        section="Advanced"
    ),
    SettingSpec(
        name="tools",
        widget_type=WidgetType.MULTI_CHOICE,
        default=["pan", "wheel_zoom", "box_zoom", "reset", "hover"],
        options=["pan", "wheel_zoom", "box_zoom", "box_select", "lasso_select", "crosshair", "hover", "reset"],
        label="Interactive Tools",
        width=300,
        section="Advanced"
    )
]
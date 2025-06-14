# battery_dashboard/ui/base_tab.py
import panel as pn
import param
from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, Any, Optional, List
import polars as pl

from ..core.data_manager import cell_data_manager, cycle_data_manager
from ..core.state_manager import AppStateManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Create a custom metaclass that inherits from both metaclasses
class ParameterizedABCMeta(ABCMeta, type(param.Parameterized)):
    pass


# Use the custom metaclass with BaseTab
class BaseTab(param.Parameterized, ABC, metaclass=ParameterizedABCMeta):
    """Base class for all dashboard tabs providing common functionality"""

    # Common parameters
    title = param.String(default="Tab", doc="Tab title")
    enabled = param.Boolean(default=True, doc="Whether tab is enabled")
    loading = param.Boolean(default=False, doc="Loading state")
    error_message = param.String(default="", doc="Current error message")

    def __init__(self, state_manager: AppStateManager, **params):
        super().__init__(**params)
        self.state_manager = state_manager
        self.container = pn.Column(sizing_mode="stretch_both")
        self.sidebar = pn.Column(width=300, sizing_mode="fixed")
        self.main_content = pn.Column(sizing_mode="stretch_both")

        # Watch state changes
        self.state_manager.param.watch(self.on_state_change, [
            'selected_cell_ids',
            'selected_cell_data',
            'cycle_data'
        ])

        # Setup the tab
        self.setup_tab()

    @abstractmethod
    def setup_tab(self):
        """Setup tab-specific components. Override in subclasses."""
        pass

    @abstractmethod
    def create_controls(self) -> pn.Column:
        """Create tab-specific control widgets. Override in subclasses."""
        pass

    @abstractmethod
    def create_main_content(self) -> pn.layout.Panel:
        """Create main content area. Override in subclasses."""
        pass

    def on_state_change(self, event):
        """Handle global state changes. Override in subclasses if needed."""
        if event.name == 'selected_cell_ids':
            self.on_cell_selection_change(event.new)
        elif event.name == 'selected_cell_data':
            self.on_cell_data_change(event.new)
        elif event.name == 'cycle_data':
            self.on_cycle_data_change(event.new)

    def on_cell_selection_change(self, cell_ids: List[int]):
        """Handle cell selection changes. Override in subclasses."""
        pass

    def on_cell_data_change(self, cell_data: Optional[pl.DataFrame]):
        """Handle cell data changes. Override in subclasses."""
        pass

    def on_cycle_data_change(self, cycle_data: Optional[pl.DataFrame]):
        """Handle cycle data changes. Override in subclasses."""
        pass

    def create_layout(self) -> pn.layout.Panel:
        """Create the standard tab layout"""
        # Update sidebar and main content
        self.sidebar.objects = [self.create_controls()]
        self.main_content.objects = [self.create_main_content()]

        # Create the layout
        layout = pn.Row(
            self.sidebar,
            self.main_content,
            sizing_mode="stretch_both"
        )

        return layout

    def show_loading(self, message: str = "Loading..."):
        """Show loading indicator"""
        self.loading = True
        self.main_content.objects = [
            pn.Column(
                pn.indicators.Progress(active=True, value=50, width=400),
                pn.pane.Markdown(f"**{message}**", styles={"text-align": "center"}),
                align="center"
            )
        ]

    def show_error(self, error: str):
        """Show error message"""
        self.loading = False
        self.error_message = error
        self.main_content.objects = [
            pn.pane.Alert(
                f"**Error:** {error}",
                alert_type="danger",
                margin=(20, 20)
            )
        ]

    def show_empty_state(self, message: str = "No data available"):
        """Show empty state message"""
        self.loading = False
        self.main_content.objects = [
            pn.pane.Markdown(
                f"*{message}*",
                styles={"color": "#666", "font-style": "italic", "text-align": "center"},
                margin=(50, 20)
            )
        ]

    def clear_error(self):
        """Clear error state"""
        self.error_message = ""

    def create_info_card(self, title: str, content: pn.layout.Panel, collapsed: bool = True) -> pn.Card:
        """Create a standardized info card"""
        return pn.Card(
            content,
            title=title,
            collapsed=collapsed,
            collapsible=True,
            header_background="#17a2b8",
            header_color="white",
            margin=(10, 0)
        )

    def create_settings_card(self, title: str, content: pn.layout.Panel, collapsed: bool = True) -> pn.Card:
        """Create a standardized settings card"""
        return pn.Card(
            content,
            title=title,
            collapsed=collapsed,
            collapsible=True,
            header_background="#3498db",
            header_color="white",
            margin=(10, 0)
        )

    def create_action_button(self, name: str, callback, button_type: str = "primary", **kwargs) -> pn.widgets.Button:
        """Create a standardized action button"""
        button = pn.widgets.Button(
            name=name,
            button_type=button_type,
            width=150,
            **kwargs
        )
        button.on_click(callback)
        return button

    def validate_data_available(self) -> bool:
        """Check if required data is available"""
        return (self.state_manager.selected_cell_ids and
                len(self.state_manager.selected_cell_ids) > 0)

    def get_selected_data(self) -> Optional[pl.DataFrame]:
        """Get currently selected cell data"""
        return self.state_manager.selected_cell_data

    def get_cycle_data(self) -> Optional[pl.DataFrame]:
        """Get current cycle data"""
        return self.state_manager.cycle_data

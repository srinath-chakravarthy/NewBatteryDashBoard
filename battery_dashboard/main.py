# battery_dashboard/main.py
import panel as pn
import param
import asyncio
from dotenv import load_dotenv

# Import config
from battery_dashboard.config import LOG_FILE, LOG_LEVEL
# Import core components
from battery_dashboard.core.state_manager import app_state
from battery_dashboard.ui.components.cell_selector import CellSelectorTab
from battery_dashboard.ui.components.cycle_plots import CyclePlotsTab
# from battery_dashboard.ui.extensions import create_extensions
from battery_dashboard.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(LOG_LEVEL, LOG_FILE)
logger = get_logger(__name__)

# Panel extensions
pn.extension("plotly", "tabulator", "modal", sizing_mode="stretch_width")
# create_extensions()

logger.info("Panel extensions loaded")


class BatteryDashboard(param.Parameterized):
    """Main dashboard application using the new architecture"""

    theme = param.Selector(default="default", objects=["default", "dark"])

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize components with state manager
        self.cell_selector_tab = CellSelectorTab(app_state)
        self.cycle_plots_tab = CyclePlotsTab(app_state)

        # Create theme toggle
        self.theme_toggle = pn.widgets.Toggle(
            name="Dark Mode",
            value=False,
            width=100,
            align="end"
        )
        self.theme_toggle.param.watch(self._on_theme_change, "value")

        # Create status indicator
        self.status_indicator = pn.pane.Markdown("", styles={"color": "white", "font-size": "0.9em"})

        # Watch app state for status updates
        app_state.param.watch(self._update_status, ["loading_states", "error_states"])

        # Initialize the application
        self._initialize_app()

    def _initialize_app(self):
        """Initialize the application asynchronously"""
        try:
            pn.state.onload(self._do_init)
        except Exception as e:
            logger.error(f"Failed to schedule initialization: {e}")

    async def _do_init(self):
        try:
            await app_state.initialize()
            self.status_indicator.object = f"✅ Loaded {len(app_state.cell_data)} cells"
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            self.status_indicator.object = f"❌ Initialization failed: {str(e)}"

    def _on_theme_change(self, event):
        """Handle theme change"""
        self.theme = "dark" if event.new else "default"
        # Update CSS variables based on theme
        if event.new:
            # Apply dark theme
            pn.config.apply_theme("dark")
        else:
            # Apply light theme
            pn.config.apply_theme("default")

    def _update_status(self, event):
        """Update status indicator based on app state"""
        if app_state.is_loading():
            operations = list(app_state.loading_states.keys())
            if "cycle_data" in operations:
                message = app_state.get_loading_message("cycle_data")
                self.status_indicator.object = f"🔄 {message}"
            else:
                self.status_indicator.object = f"🔄 Loading ({', '.join(operations)})"
        elif app_state.has_error():
            error_ops = list(app_state.error_states.keys())
            self.status_indicator.object = f"❌ Error in {', '.join(error_ops)}"
        else:
            # Show summary
            summary = app_state.get_state_summary()
            if summary['selected_cells'] > 0:
                self.status_indicator.object = f"✅ {summary['selected_cells']} cells selected"
            elif summary['cell_data_loaded']:
                self.status_indicator.object = f"✅ {summary['cell_data_rows']} cells loaded"
            else:
                self.status_indicator.object = "⚠️ No data loaded"

    def create_layout(self):
        """Create the main application layout"""

        # Create header with branding and controls
        header = pn.Row(
            pn.pane.Markdown(
                "# 🔋 Battery Analytics Dashboard",
                styles={"color": "white", "margin-bottom": "0px", "font-weight": "bold"}
            ),
            pn.Spacer(),
            self.status_indicator,
            self.theme_toggle,
            height=70,
            styles={
                "padding": "0 20px",
                "background": "linear-gradient(90deg, #1e3c72 0%, #2a5298 100%)",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            align="center"
        )

        # Create navigation tabs
        tabs = pn.Tabs(
            ("📊 Cell Selector", self.cell_selector_tab.create_layout()),
            ("📈 Cycle Analysis", self.cycle_plots_tab.create_layout()),
            ("📉 Statistics", self._create_placeholder_tab("Statistics", "Statistical analysis coming soon...")),
            ("🤖 ML Analysis", self._create_placeholder_tab("ML Analysis", "Machine learning features coming soon...")),
            dynamic=True,
            margin=(10, 0),
            styles={"min-height": "calc(100vh - 120px)"}
        )

        # Create footer with version and cache info
        footer = pn.Row(
            pn.pane.Markdown(
                "**Battery Analytics Dashboard v2.0** | "
                f"Cells: {len(app_state.cell_data) if hasattr(app_state, 'cell_data') 
                          and app_state.cell_data is not None 
                          and not app_state.cell_data.is_empty() else 0} | "
                "Built with Panel + Polars + MLflow",
                styles={"color": "#64748B", "font-size": "0.85em"}
            ),
            pn.Spacer(),
            pn.widgets.Button(
                name="Clear Cache",
                button_type="light",
                width=100,
                height=25,
                styles={"font-size": "0.8em"}
            ),
            height=40,
            styles={
                "padding": "0 20px",
                "background": "#F8FAFC",
                "border-top": "1px solid #E2E8F0"
            },
            align="center"
        )

        # Setup footer button
        footer[2].on_click(self._clear_cache)

        # Create main template
        template = pn.template.FastListTemplate(
            title="Battery Analytics",
            header=header,
            main=pn.Column(
                tabs,
                footer,
                sizing_mode="stretch_width"
            ),
            sidebar=None,
            accent_base_color="#2563EB",
            header_background="transparent",  # We handle header styling ourselves
            sidebar_width=0,
            main_max_width="100%",
            theme=self.theme
        )

        return template

    def _create_placeholder_tab(self, title: str, message: str) -> pn.layout.Panel:
        """Create a placeholder tab for future features"""
        return pn.Column(
            pn.pane.Markdown(f"# {title}", styles={"text-align": "center", "margin": "50px 0 20px 0"}),
            pn.pane.Markdown(
                f"*{message}*",
                styles={"text-align": "center", "color": "#666", "font-size": "1.1em"}
            ),
            pn.pane.Markdown(
                "This feature is planned for a future release. "
                "The modular architecture makes it easy to add new analysis capabilities.",
                styles={"text-align": "center", "margin": "20px auto", "max-width": "500px"}
            ),
            pn.Spacer(),
            sizing_mode="stretch_width"
        )

    def _clear_cache(self, event):
        """Clear application cache"""
        try:
            from battery_dashboard.core.data_manager import cell_data_manager, cycle_data_manager

            # Clear data manager caches
            cell_data_manager.cache.clear()
            cycle_data_manager.cache.clear()

            # Clear app state cache
            app_state.clear_analysis_cache()

            # Update status
            self.status_indicator.object = "🧹 Cache cleared"

            logger.info("Application cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            self.status_indicator.object = f"❌ Error clearing cache: {str(e)}"


# Create and serve the dashboard
def create_app():
    """Create the dashboard application"""
    main_dashboard = BatteryDashboard()
    return main_dashboard.create_layout()


# For panel serve
if __name__ == "__main__":
    app = create_app()
    app.servable()
else:
    # For imports
    dashboard = BatteryDashboard()
    app = dashboard.create_layout()
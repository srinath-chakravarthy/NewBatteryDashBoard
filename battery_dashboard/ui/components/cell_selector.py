# battery_dashboard/ui/components/cell_selector.py
import panel as pn
import polars as pl
import param
import re
from typing import List, Optional, Dict, Any
import asyncio

from ..base_tab import BaseTab
from ...core.state_manager import AppStateManager
from ...core.data_manager import cell_data_manager
from ...utils.logging import get_logger

logger = get_logger(__name__)


def create_filter_widgets(df: pl.DataFrame) -> Dict[str, pn.widgets.Select]:
    """Dynamically generate filter widgets based on available columns, handling NaN values."""
    filter_widgets = {}
    for column in ["design_name", "experiment_group", "layer_types", "test_status", "test_year"]:
        if column in df.columns:
            options = df[column].drop_nulls().unique().to_list()
            options = sorted([str(opt) for opt in options])  # Ensure all values are strings
            options = [""] + options  # Add empty option for 'no filter'
            filter_widgets[column] = pn.widgets.Select(
                name=column.replace("_", " ").title(),
                options=options,
                width=200
            )
    return filter_widgets


def apply_filters(data: pl.DataFrame, filters: Dict[str, pn.widgets.Select]) -> pl.DataFrame:
    """Apply filter widget selections to the Polars dataframe."""
    filtered_data = data.clone()
    for key, widget in filters.items():
        if widget.value:
            filtered_data = filtered_data.filter(pl.col(key) == widget.value)
    return filtered_data


class CellSelectorTab(BaseTab):
    """Refactored Cell Selector Tab using new architecture"""

    def __init__(self, state_manager: AppStateManager, **params):
        super().__init__(state_manager, title="Cell Selector", **params)

        # Internal state
        self.filtered_cell_data = None
        self.search_query = ""

        # Required and display columns
        self.required_columns = ["cell_id"]
        self.default_columns = [
            "cell_id", "cell_name", "actual_nominal_capacity_ah", "regular_cycles",
            "last_discharge_capacity", "discharge_capacity_retention"
        ]

    def setup_tab(self):
        """Setup tab-specific components"""
        # Initialize with loading state
        # Missing widgets that are referenced in methods:
        self.search_input = pn.widgets.TextInput(placeholder="Loading...")
        self.search_button = pn.widgets.Button(name="Search", disabled=True)
        self.clear_search_button = pn.widgets.Button(name="Clear", disabled=True)
        self.search_info = pn.pane.Markdown("")

        self.data_table = pn.widgets.Tabulator(value=pl.DataFrame().to_pandas())
        self.column_selector = pn.widgets.MultiSelect(name="Columns", options=[])

        self.selection_indicator = pn.pane.Markdown("**0** cells selected")
        self.select_all_button = pn.widgets.Button(name="Select All", disabled=True)
        self.clear_selection_button = pn.widgets.Button(name="Clear Selection", disabled=True)
        self.load_button = pn.widgets.Button(name="Load Cycle Data", disabled=True)

        # Already identified:
        self.stats_content = pn.Column()
        self.filter_widgets = {}

        self.show_loading("Loading cell data...")

        # Setup will complete when cell data is loaded via state manager

    def on_cell_data_change(self, cell_data: Optional[pl.DataFrame]):
        """Handle cell data changes from state manager"""
        if cell_data is None or cell_data.is_empty():
            self.show_empty_state("No cell data available.")
            logger.info("No cell data available.")
            return

        logger.info(f"Received cell data: {len(cell_data)} rows")

        # Store the cell data
        self.cell_data = cell_data
        self.filtered_cell_data = cell_data

        # Create UI components now that we have data
        self.create_search_components()
        self.create_filter_widgets()
        self.create_table_components()
        self.create_action_components()
        self.create_statistics_components()

        # Setup event handlers
        self.setup_event_handlers()

        # Initial table update
        self.update_table_data()

        # Update the main content
        self.container.objects = [self.create_main_table_area()]

    def create_search_components(self):
        """Create search UI components"""
        self.search_input = pn.widgets.TextInput(
            name="Search",
            placeholder="Enter search terms (column:value or free text)",
            width=400
        )
        self.search_button = pn.widgets.Button(
            name="Search",
            button_type="primary",
            width=100
        )
        self.clear_search_button = pn.widgets.Button(
            name="Clear",
            button_type="default",
            width=100
        )
        self.search_info = pn.pane.Markdown("", styles={"color": "blue", "font-style": "italic"})

    def create_filter_widgets(self):
        """Create filter widgets based on available data"""
        self.filter_widgets = create_filter_widgets(self.cell_data)

    def create_table_components(self):
        """Create table and column selection components"""
        # Get optional columns (excluding required ones)
        self.optional_columns = [col for col in self.cell_data.columns if col not in self.required_columns]

        # Column selector
        self.column_selector = pn.widgets.MultiSelect(
            name="Select Columns to Display",
            options=self.optional_columns,
            value=[col for col in self.default_columns if col not in self.required_columns],
            size=6,
            width=300
        )

        # Main data table
        self.data_table = pn.widgets.Tabulator(
            pagination="remote",
            page_size=50,
            selectable="checkbox",
            styles={
                '_selector': {'padding': '12px'},
                '*': {'padding': '4px'}
            },
            header_align="left",
            theme="bootstrap4",
            layout="fit_data_table",
            show_index=False,
            frozen_rows=[-2, -1],
            height=600,
            theme_classes=["table-bordered", "thead-dark"],
        )

    def create_action_components(self):
        """Create action buttons and selection indicators"""
        # Selection indicator
        self.selection_indicator = pn.pane.Markdown("**0** cells selected")

        # Selection buttons
        self.select_all_button = pn.widgets.Button(
            name="Select All",
            button_type="primary",
            width=120,
            icon="check-square"
        )

        self.clear_selection_button = pn.widgets.Button(
            name="Clear Selection",
            button_type="default",
            width=120,
            icon="square"
        )

        # Load data button
        self.load_button = pn.widgets.Button(
            name="Load Cycle Data",
            button_type="primary",
            width=200,
            disabled=True
        )

    def create_statistics_components(self):
        """Create statistics display components"""
        self.stats_content = pn.Column(
            pn.pane.Markdown("### Select cells to view statistics", styles={"color": "#666"})
        )

    def setup_event_handlers(self):
        """Setup all event handlers"""
        # Filter widgets
        for widget in self.filter_widgets.values():
            widget.param.watch(self.update_table_data, "value")

        # Table selection
        self.data_table.param.watch(self.on_cell_selection, "selection")

        # Column selector
        self.column_selector.param.watch(self.update_table_data, "value")

        # Search components
        self.search_button.on_click(self.on_search)
        self.clear_search_button.on_click(self.clear_search)
        self.search_input.param.watch(self.on_search_enter, "value_input")

        # Action buttons
        self.select_all_button.on_click(self.select_all_cells)
        self.clear_selection_button.on_click(self.clear_selection)
        self.load_button.on_click(self.trigger_data_loading)

    def create_controls(self) -> pn.Column:
        """Create sidebar controls"""
        return pn.Column(
            pn.pane.Markdown("## Filters"),
            *list(self.filter_widgets.values()),
            pn.layout.Divider(),
            pn.pane.Markdown("## Display Settings"),
            self.column_selector,
            pn.layout.Divider(),
            self.create_statistics_card(),
            width=300
        )

    def create_main_content(self) -> pn.layout.Panel:
        """Create main content area"""
        return pn.Column(
            pn.pane.Markdown("# Cell Selection", styles={"margin-bottom": "10px"}),
            self.container,
            sizing_mode="stretch_both"
        )

    def create_main_table_area(self) -> pn.layout.Panel:
        """Create the main table area with search and actions"""
        # Search row
        search_row = pn.Row(
            self.search_input,
            self.search_button,
            self.clear_search_button,
            styles={"margin-bottom": "10px"}
        )

        # Selection buttons row
        selection_buttons = pn.Row(
            self.select_all_button,
            self.clear_selection_button,
            align="center",
            styles={"margin-bottom": "10px"}
        )

        # Table in scrollable container
        table_container = pn.Column(
            self.data_table,
            max_height=650,
            scroll=True,
            width_policy='max',
            styles={'overflow-y': 'auto', 'overflow-x': 'auto'}
        )

        # Action row below table
        action_row = pn.Row(
            self.load_button,
            self.selection_indicator,
            align="center",
            styles={"margin-top": "10px"}
        )

        return pn.Column(
            search_row,
            self.search_info,
            selection_buttons,
            table_container,
            action_row,
            sizing_mode="stretch_width"
        )

    def create_statistics_card(self) -> pn.Card:
        """Create statistics card for sidebar"""
        return pn.Card(
            self.stats_content,
            title="Selection Statistics",
            collapsed=False,
            collapsible=True,
            header_background="#3b82f6",
            header_color="white",
            margin=(0, 0, 10, 0)
        )

    def on_search_enter(self, event):
        """Handle pressing Enter in the search box"""
        if event.new and event.new.endswith('\n'):
            self.search_input.value = event.new.rstrip('\n')
            self.on_search(None)

    def on_search(self, event):
        """Execute the search query"""
        query = self.search_input.value.strip()
        self.search_query = query
        self.update_table_data()

    def clear_search(self, event):
        """Clear the search query and reset the table"""
        self.search_input.value = ""
        self.search_query = ""
        self.search_info.object = ""
        self.update_table_data()

    def apply_search_query(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the search query to the dataframe"""
        if not self.search_query:
            return df

        query = self.search_query.lower()

        # Parse column:value patterns
        column_search_pattern = r'(\w+):\s*([\w.%<>=]+)'
        column_searches = re.findall(column_search_pattern, query)

        filtered_df = df
        search_applied = False

        # Handle column-specific searches
        for col_name, search_value in column_searches:
            query = query.replace(f"{col_name}:{search_value}", "").strip()

            # Find actual column name (case-insensitive)
            actual_col = next((c for c in df.columns if c.lower() == col_name.lower()), None)

            if actual_col is None:
                self.search_info.object = f"⚠️ Column '{col_name}' not found"
                continue

            # Handle comparison operators
            if any(op in search_value for op in ['>', '<', '=']):
                try:
                    if '>=' in search_value:
                        val = float(search_value.replace('>=', '').strip())
                        filtered_df = filtered_df.filter(pl.col(actual_col) >= val)
                    elif '<=' in search_value:
                        val = float(search_value.replace('<=', '').strip())
                        filtered_df = filtered_df.filter(pl.col(actual_col) <= val)
                    elif '>' in search_value:
                        val = float(search_value.replace('>', '').strip())
                        filtered_df = filtered_df.filter(pl.col(actual_col) > val)
                    elif '<' in search_value:
                        val = float(search_value.replace('<', '').strip())
                        filtered_df = filtered_df.filter(pl.col(actual_col) < val)
                    elif '=' in search_value:
                        val_str = search_value.replace('=', '').strip()
                        try:
                            val = float(val_str)
                            filtered_df = filtered_df.filter(pl.col(actual_col) == val)
                        except ValueError:
                            filtered_df = filtered_df.filter(
                                pl.col(actual_col).cast(pl.Utf8).str.contains(val_str, literal=True)
                            )
                    search_applied = True
                except ValueError:
                    self.search_info.object = f"⚠️ Invalid numeric value in '{search_value}'"
            else:
                # Simple text match
                filtered_df = filtered_df.filter(
                    pl.col(actual_col).cast(pl.Utf8).str.contains(search_value, literal=True)
                )
                search_applied = True

        # Apply free text search across all string columns
        query = query.strip()
        if query:
            string_filters = []
            for col in df.columns:
                try:
                    string_filters.append(pl.col(col).cast(pl.Utf8).str.contains(query))
                except:
                    continue

            if string_filters:
                combined_filter = string_filters[0]
                for filter_expr in string_filters[1:]:
                    combined_filter = combined_filter | filter_expr

                filtered_df = filtered_df.filter(combined_filter)
                search_applied = True

        if search_applied:
            rows_found = len(filtered_df)
            total_rows = len(df)
            self.search_info.object = f"Found {rows_found} of {total_rows} cells matching search criteria"

        return filtered_df

    def update_table_data(self, *events):
        """Update table data based on current filters and search"""
        # Apply row filters
        self.filtered_cell_data = apply_filters(self.cell_data, self.filter_widgets)

        # Apply search query
        self.filtered_cell_data = self.apply_search_query(self.filtered_cell_data)

        # Get selected columns for display
        selected_columns = self.column_selector.value or []
        display_columns = self.required_columns + [
            col for col in selected_columns if col not in self.required_columns
        ]

        # Create display DataFrame
        display_df = self.filtered_cell_data.select(display_columns).to_pandas()

        # Update the table
        self.data_table.value = display_df

        # Clear selection when filters change
        self.data_table.selection = []
        self.update_selection_display([])

        logger.info(f"Table updated with {len(display_df)} rows after filtering")

    def select_all_cells(self, event):
        """Select all cells in the current filtered dataset"""
        if self.data_table.value is not None and not self.data_table.value.empty:
            all_indices = list(range(len(self.data_table.value)))
            self.data_table.selection = all_indices

    def clear_selection(self, event):
        """Clear the current selection"""
        self.data_table.selection = []

    def on_cell_selection(self, event):
        """Handle cell selection changes"""
        selected_indices = event.new if hasattr(event, "new") else []
        self.update_selection_display(selected_indices)

    def update_selection_display(self, selected_indices: List[int]):
        """Update selection display and statistics"""
        if not selected_indices:
            selected_cell_ids = []
            selected_data = None
        else:
            # Get selected cell_ids from displayed table
            selected_rows = self.data_table.value.iloc[selected_indices]
            selected_cell_ids = selected_rows['cell_id'].tolist()

            # Get full data for selected cells
            selected_data = self.filtered_cell_data.filter(
                pl.col("cell_id").is_in(selected_cell_ids)
            )

        # Update selection indicator
        self.selection_indicator.object = f"**{len(selected_cell_ids)}** cells selected"

        # Update load button state
        self.update_load_button_state(selected_cell_ids)

        # Update statistics
        self.update_selection_statistics(selected_data)

        # Store for data loading
        self.selected_cell_ids = selected_cell_ids
        self.selected_data = selected_data

        logger.info(f"Selection updated: {len(selected_cell_ids)} cells")

    def update_load_button_state(self, selected_cell_ids: List[int]):
        """Update the load button state based on selection"""
        if selected_cell_ids:
            self.load_button.disabled = False
            count = len(selected_cell_ids)
            button_text = f"Load Cycle Data ({count})"

            if count > 20:
                button_text += " ⚠️"
                self.load_button.tooltip = f"Loading {count} cells may take some time"
            else:
                self.load_button.tooltip = ""

            self.load_button.name = button_text
        else:
            self.load_button.disabled = True
            self.load_button.name = "Load Cycle Data"
            self.load_button.tooltip = "Select cells first"

    def update_selection_statistics(self, selected_data: Optional[pl.DataFrame]):
        """Update the statistics card with information about selected cells"""
        self.stats_content.clear()

        if selected_data is None or selected_data.is_empty():
            self.stats_content.append(
                pn.pane.Markdown("### Select cells to view statistics", styles={"color": "#666"})
            )
            return

        num_cells = len(selected_data)
        stats_md = f"### {num_cells} Cells Selected\n\n"

        # Calculate statistical summaries for key numeric columns
        key_metrics = [
            "actual_nominal_capacity_ah", "last_discharge_capacity",
            "discharge_capacity_retention", "total_cycles"
        ]

        summary_stats = []

        for metric in key_metrics:
            if metric in selected_data.columns:
                try:
                    metric_data = selected_data.select(pl.col(metric)).drop_nulls()
                    if not metric_data.is_empty():
                        mean_val = metric_data.mean().item()
                        min_val = metric_data.min().item()
                        max_val = metric_data.max().item()
                        std_val = metric_data.std().item()

                        display_name = metric.replace("_", " ").title()
                        summary_stats.append(f"**{display_name}**")
                        summary_stats.append(f"- Mean: {mean_val:.2f}")
                        summary_stats.append(f"- Min: {min_val:.2f}")
                        summary_stats.append(f"- Max: {max_val:.2f}")
                        summary_stats.append(f"- Std Dev: {std_val:.2f}")
                        summary_stats.append("")
                except Exception as e:
                    logger.warning(f"Error calculating stats for {metric}: {str(e)}")

        if summary_stats:
            stats_md += "\n".join(summary_stats)
        else:
            stats_md += "No numeric data available for statistics."

        self.stats_content.append(pn.pane.Markdown(stats_md))

    def trigger_data_loading(self, event):
        """Trigger cycle data loading through state manager"""
        if not hasattr(self, 'selected_cell_ids') or not self.selected_cell_ids:
            logger.warning("No cells selected for data loading")
            return

        logger.info(f"Triggering cycle data loading for {len(self.selected_cell_ids)} cells")

        # Update state manager asynchronously
        async def load_data():
            await self.state_manager.update_cell_selection(
                self.selected_cell_ids,
                self.selected_data
            )

        # Schedule the async operation
        pn.state.schedule_callback(load_data())
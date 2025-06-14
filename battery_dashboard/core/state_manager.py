# battery_dashboard/core/state_manager.py
import param
import polars as pl
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from .data_manager import cell_data_manager, cycle_data_manager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AppStateManager(param.Parameterized):
    """Centralized application state management"""

    # Data state
    cell_data = param.Parameter(default=None, doc="Full cell dataset")
    selected_cell_ids = param.List(default=[], doc="Currently selected cell IDs")
    selected_cell_data = param.Parameter(default=None, doc="Data for selected cells")
    cycle_data = param.Parameter(default=None, doc="Cycle data for selected cells")

    # UI state
    active_tab = param.String(default="cell_selector", doc="Currently active tab")
    loading_states = param.Dict(default={}, doc="Loading states for different operations")
    error_states = param.Dict(default={}, doc="Error states for different operations")

    # Filter state
    applied_filters = param.Dict(default={}, doc="Currently applied filters")
    search_query = param.String(default="", doc="Current search query")

    # Analysis state
    analysis_results = param.Dict(default={}, doc="Cached analysis results")
    plot_configs = param.Dict(default={}, doc="Plot configurations by tab")

    def __init__(self, **params):
        super().__init__(**params)
        self._initialization_complete = False

    async def initialize(self):
        """Initialize the state manager and load initial data"""
        logger.info("Initializing application state...")

        try:
            # Set loading state
            self.set_loading_state("cell_data", True, "Loading initial cell data...")

            # Load initial cell data
            cell_data = await cell_data_manager.load_initial_data()
            with param.edit_constant(self):
                self.cell_data = cell_data

            # Clear loading state
            self.set_loading_state("cell_data", False)

            self._initialization_complete = True
            logger.info(f"Application state initialized with {len(cell_data)} cells")

        except Exception as e:
            logger.error(f"Error initializing application state: {e}")
            self.set_error_state("initialization", str(e))
            raise

    def set_loading_state(self, operation: str, is_loading: bool, message: str = ""):
        """Set loading state for a specific operation"""
        current_states = self.loading_states.copy()
        if is_loading:
            current_states[operation] = {
                'loading': True,
                'message': message,
                'started_at': datetime.now()
            }
        else:
            if operation in current_states:
                del current_states[operation]

        self.loading_states = current_states

    def set_error_state(self, operation: str, error_message: str):
        """Set error state for a specific operation"""
        current_errors = self.error_states.copy()
        current_errors[operation] = {
            'error': error_message,
            'timestamp': datetime.now()
        }
        self.error_states = current_errors

    def clear_error_state(self, operation: str):
        """Clear error state for a specific operation"""
        current_errors = self.error_states.copy()
        if operation in current_errors:
            del current_errors[operation]
            self.error_states = current_errors

    def is_loading(self, operation: str = None) -> bool:
        """Check if an operation is currently loading"""
        if operation:
            return operation in self.loading_states
        return len(self.loading_states) > 0

    def get_loading_message(self, operation: str) -> str:
        """Get loading message for a specific operation"""
        if operation in self.loading_states:
            return self.loading_states[operation].get('message', 'Loading...')
        return ""

    def has_error(self, operation: str = None) -> bool:
        """Check if there's an error for a specific operation"""
        if operation:
            return operation in self.error_states
        return len(self.error_states) > 0

    def get_error_message(self, operation: str) -> str:
        """Get error message for a specific operation"""
        if operation in self.error_states:
            return self.error_states[operation].get('error', '')
        return ""

    async def update_cell_selection(self, cell_ids: List[int], cell_data: Optional[pl.DataFrame] = None):
        """Update selected cells and trigger data loading"""
        logger.info(f"Updating cell selection: {len(cell_ids)} cells")

        # Update selection
        self.selected_cell_ids = cell_ids
        self.selected_cell_data = cell_data

        # Clear previous cycle data and errors
        self.cycle_data = None
        self.clear_error_state("cycle_data")

        if not cell_ids:
            return

        try:
            # Set loading state
            self.set_loading_state("cycle_data", True, f"Loading cycle data for {len(cell_ids)} cells...")

            # Load cycle data asynchronously
            cycle_data = await cycle_data_manager.get_cycle_data(
                cell_ids=cell_ids,
                cell_metadata=cell_data,
                progress_callback=self._update_cycle_loading_progress
            )

            # Update state
            self.cycle_data = cycle_data
            self.set_loading_state("cycle_data", False)

            logger.info(f"Loaded cycle data: {len(cycle_data)} rows")

        except Exception as e:
            logger.error(f"Error loading cycle data: {e}")
            self.set_error_state("cycle_data", str(e))
            self.set_loading_state("cycle_data", False)

    def _update_cycle_loading_progress(self, progress: float, message: str):
        """Update cycle loading progress"""
        current_states = self.loading_states.copy()
        if "cycle_data" in current_states:
            current_states["cycle_data"]["message"] = f"{message} ({progress:.0%})"
            self.loading_states = current_states

    def apply_filters(self, filters: Dict[str, Any]):
        """Apply filters to cell data"""
        logger.info(f"Applying filters: {filters}")
        self.applied_filters = filters

        if self.cell_data is not None:
            # Apply filters using data manager
            filtered_data = cell_data_manager.get_filtered_data(filters)
            # Note: This doesn't update selected_cell_data directly
            # The UI components should react to applied_filters changes

    def apply_search(self, query: str):
        """Apply search query to cell data"""
        logger.info(f"Applying search: {query}")
        self.search_query = query

        if self.cell_data is not None:
            # Apply search using data manager
            searched_data = cell_data_manager.search_cells(query)
            # Note: This doesn't update selected_cell_data directly
            # The UI components should react to search_query changes

    def save_plot_config(self, tab_name: str, config: Dict[str, Any]):
        """Save plot configuration for a specific tab"""
        current_configs = self.plot_configs.copy()
        current_configs[tab_name] = config
        self.plot_configs = current_configs

    def get_plot_config(self, tab_name: str) -> Optional[Dict[str, Any]]:
        """Get plot configuration for a specific tab"""
        return self.plot_configs.get(tab_name)

    def cache_analysis_result(self, analysis_type: str, cell_ids: List[int], result: Any):
        """Cache analysis result"""
        cache_key = f"{analysis_type}_{hash(tuple(sorted(cell_ids)))}"
        current_results = self.analysis_results.copy()
        current_results[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'cell_ids': cell_ids
        }
        self.analysis_results = current_results

    def get_cached_analysis_result(self, analysis_type: str, cell_ids: List[int]) -> Optional[Any]:
        """Get cached analysis result"""
        cache_key = f"{analysis_type}_{hash(tuple(sorted(cell_ids)))}"
        if cache_key in self.analysis_results:
            cached = self.analysis_results[cache_key]
            # Check if cache is recent (within 1 hour)
            if (datetime.now() - cached['timestamp']).total_seconds() < 3600:
                return cached['result']
        return None

    def clear_analysis_cache(self):
        """Clear all cached analysis results"""
        self.analysis_results = {}

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current application state"""
        return {
            'cell_data_loaded': self.cell_data is not None,
            'cell_data_rows': len(self.cell_data) if self.cell_data is not None else 0,
            'selected_cells': len(self.selected_cell_ids),
            'cycle_data_loaded': self.cycle_data is not None,
            'cycle_data_rows': len(self.cycle_data) if self.cycle_data is not None else 0,
            'active_tab': self.active_tab,
            'applied_filters': self.applied_filters,
            'search_query': self.search_query,
            'loading_operations': list(self.loading_states.keys()),
            'error_operations': list(self.error_states.keys()),
            'cached_analyses': len(self.analysis_results),
            'initialization_complete': self._initialization_complete
        }


# Global state manager instance
app_state = AppStateManager()
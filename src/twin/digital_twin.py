"""
twin/digital_twin.py

Main entry point for the AI Digital Twin system. Wires together every
component — data loading, spatial/temporal reasoning, state tracking,
querying, and simulation — behind a single, simple public interface.
This is the only class most external code (notebooks, future APIs,
the future Streamlit dashboard) should need to import directly.
"""

from pathlib import Path

import xarray as xr

from src.ingestion.imd_loader import IMDLoader

from src.twin.spatial_engine import SpatialEngine
from src.twin.temporal_engine import TemporalEngine
from src.twin.state_manager import StateManager, DigitalTwinState
from src.twin.query_engine import QueryEngine, QueryResult
from src.twin.simulation_engine import SimulationEngine

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DigitalTwin:
    """
    The AI Digital Twin: a single, unified interface over the full
    rainfall data and simulation system.

    Single Responsibility:
        Composition and public interface only.

    Architecture:
        User
          -> DigitalTwin
               -> SpatialEngine   (where)
               -> TemporalEngine  (when)
               -> StateManager    (remembers last query)
               -> QueryEngine     (combines spatial + temporal + state)
               -> SimulationEngine (what-if scenarios)
          -> IMD Rainfall Dataset

    Attributes:
        dataset_path (Path): Path to the source NetCDF dataset.
        dataset (xr.Dataset): The loaded, raw rainfall dataset.
        spatial (SpatialEngine): Handles location-based queries.
        temporal (TemporalEngine): Handles date/time-based queries.
        state (StateManager): Tracks the twin's current state.
        query (QueryEngine): Combines spatial + temporal + state.
        simulation (SimulationEngine): Runs what-if scenarios.
    """

    def __init__(self, dataset_path: str | Path) -> None:
        """
        Construct the Digital Twin: load the dataset and build every
        internal component.

        Args:
            dataset_path: Path to the source IMD rainfall NetCDF
                file.

        Raises:
            DatasetNotFoundError: If no file exists at dataset_path.
            CoordinateNotFoundError: If the dataset is missing a
                recognizable latitude, longitude, or time coordinate.
        """
        self.dataset_path: Path = Path(dataset_path)

        logger.info(f"Constructing DigitalTwin from: {self.dataset_path}")

        loader = IMDLoader(self.dataset_path)
        self.dataset: xr.Dataset = loader.load()

        self.spatial: SpatialEngine = SpatialEngine(self.dataset)
        self.temporal: TemporalEngine = TemporalEngine(self.dataset)
        self.state: StateManager = StateManager()

        self.query: QueryEngine = QueryEngine(
            self.spatial,
            self.temporal,
            self.state
        )

        # SimulationEngine now receives time_name explicitly from
        # TemporalEngine, rather than re-detecting it independently.
        # This guarantees all three engines agree on the same
        # coordinate name — closing the item flagged since Day 1.
        self.simulation: SimulationEngine = SimulationEngine(
            self.dataset,
            time_name=self.temporal.time_name
        )

        logger.info("DigitalTwin constructed successfully — all components ready")

    def rainfall(self, latitude: float, longitude: float, date: str) -> QueryResult:
        """
        Query rainfall at a location and date.

        Raises:
            InvalidCoordinateError: If latitude/longitude is out of
                range (propagated from QueryEngine).
            DatasetSchemaError: If the rainfall variable is missing
                (propagated from QueryEngine).
            InvalidDateError: If the date is not present in the
                dataset (propagated from QueryEngine).
        """
        return self.query.rainfall_query(latitude, longitude, date)

    def current_state(self) -> DigitalTwinState:
        """Return the twin's current state."""
        return self.query.current_state()

    def reset_state(self) -> None:
        """Clear the twin's state back to its initial values."""
        self.query.reset()

    def simulate_increase(self, percentage: float) -> xr.Dataset:
        """
        Simulate a percentage increase in rainfall.

        Raises:
            SimulationError: If percentage is negative.
        """
        return self.simulation.rainfall_increase(percentage)

    def simulate_decrease(self, percentage: float) -> xr.Dataset:
        """
        Simulate a percentage decrease in rainfall.

        Raises:
            SimulationError: If percentage is negative or exceeds
                config.settings.MAX_RAINFALL_DECREASE_PERCENTAGE.
        """
        return self.simulation.rainfall_decrease(percentage)

    def simulate_dry_spell(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Simulate a dry spell (zero rainfall) over a date range.

        Raises:
            DatasetSchemaError: If the rainfall variable or time
                dimension is missing.
        """
        return self.simulation.dry_spell(start_date, end_date)

    def simulate_heavy_rainfall(
        self, start_date: str, end_date: str, multiplier: float
    ) -> xr.Dataset:
        """
        Simulate heavy rainfall over a date range.

        Raises:
            SimulationError: If multiplier is negative.
            DatasetSchemaError: If the rainfall variable or time
                dimension is missing.
        """
        return self.simulation.heavy_rainfall(start_date, end_date, multiplier)

    def reset_simulation(self) -> None:
        """Discard all simulated changes, restoring original data."""
        self.simulation.reset()
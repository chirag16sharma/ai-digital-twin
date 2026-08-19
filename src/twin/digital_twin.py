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
        Composition and public interface only. DigitalTwin builds and
        owns every other component (SpatialEngine, TemporalEngine,
        StateManager, QueryEngine, SimulationEngine) and exposes a
        small, friendly set of methods on top of them. It contains no
        domain logic of its own — every public method here is a thin
        delegate to one of its components.

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
        query (QueryEngine): Combines spatial + temporal + state for
            rainfall_query().
        simulation (SimulationEngine): Runs what-if rainfall
            scenarios on a separate, deep-copied dataset.
    """

    def __init__(self, dataset_path: str | Path) -> None:
        """
        Construct the Digital Twin: load the dataset and build every
        internal component.

        Args:
            dataset_path: Path to the source IMD rainfall NetCDF
                file.

        Raises:
            DatasetNotFoundError: If no file exists at dataset_path
                (propagated from IMDLoader.load()).
            CoordinateNotFoundError: If the dataset is missing a
                recognizable latitude, longitude, or time coordinate
                (propagated from SpatialEngine/TemporalEngine
                construction). Not wrapped into a single top-level
                exception — see the design note above this class for
                why, and reconsider if a unified
                "DigitalTwinInitializationError" would be preferred.
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

        # NOTE: SimulationEngine still hardcodes "TIME" internally
        # rather than receiving self.temporal.time_name. Flagged
        # since Day 1/2, still open — fixing it means changing
        # SimulationEngine's constructor signature, which is a
        # design change beyond today's logging/exceptions scope.
        self.simulation: SimulationEngine = SimulationEngine(
            self.dataset
        )

        logger.info("DigitalTwin constructed successfully — all components ready")

    def rainfall(
        self,
        latitude: float,
        longitude: float,
        date: str
    ) -> QueryResult:
        """
        Query rainfall at a location and date.

        Args:
            latitude: Requested latitude.
            longitude: Requested longitude.
            date: Date string, e.g. "2025-07-15".

        Returns:
            QueryResult: The resolved grid latitude/longitude, the
                queried date, and the rainfall value in mm.

        Raises:
            InvalidCoordinateError: If latitude/longitude is outside
                the dataset's coverage (propagated from QueryEngine).
            DatasetSchemaError: If "RAINFALL" is missing (propagated
                from QueryEngine).
            InvalidDateError: If the date is not present in the
                dataset (propagated from QueryEngine).
        """
        return self.query.rainfall_query(
            latitude,
            longitude,
            date
        )

    def current_state(self) -> DigitalTwinState:
        """
        Return the twin's current state (last-queried location,
        date, rainfall, and when it was last updated).

        Returns:
            DigitalTwinState: The current state dict.
        """
        return self.query.current_state()

    def reset_state(self) -> None:
        """
        Clear the twin's state back to its initial (all-None) values.

        Returns:
            None.
        """
        self.query.reset()

    def simulate_increase(self, percentage: float) -> xr.Dataset:
        """
        Simulate a percentage increase in rainfall across the entire
        simulated dataset.

        Args:
            percentage: Percentage increase to apply, e.g. 20 for a
                20% increase. Must be non-negative.

        Returns:
            xr.Dataset: The updated simulated dataset.

        Raises:
            SimulationError: If percentage is negative (propagated
                from SimulationEngine).
        """
        return self.simulation.rainfall_increase(
            percentage
        )

    def simulate_decrease(self, percentage: float) -> xr.Dataset:
        """
        Simulate a percentage decrease in rainfall across the entire
        simulated dataset.

        Args:
            percentage: Percentage decrease to apply, e.g. 20 for a
                20% decrease. Must be in [0, 100].

        Returns:
            xr.Dataset: The updated simulated dataset.

        Raises:
            SimulationError: If percentage is negative or exceeds
                100 (propagated from SimulationEngine).
        """
        return self.simulation.rainfall_decrease(
            percentage
        )

    def simulate_dry_spell(
        self,
        start_date: str,
        end_date: str
    ) -> xr.Dataset:
        """
        Simulate a dry spell (zero rainfall) over a date range.

        Args:
            start_date: Start of the dry spell, e.g. "2025-07-01".
            end_date: End of the dry spell, e.g. "2025-07-10".

        Returns:
            xr.Dataset: The updated simulated dataset.

        Raises:
            DatasetSchemaError: If "RAINFALL" or "TIME" is missing
                (propagated from SimulationEngine).
        """
        return self.simulation.dry_spell(
            start_date,
            end_date
        )

    def simulate_heavy_rainfall(
        self,
        start_date: str,
        end_date: str,
        multiplier: float
    ) -> xr.Dataset:
        """
        Simulate heavy rainfall (rainfall multiplied by a factor)
        over a date range.

        Args:
            start_date: Start of the heavy rainfall period.
            end_date: End of the heavy rainfall period.
            multiplier: Factor to multiply rainfall by, e.g. 2.0 to
                double it. Must be non-negative.

        Returns:
            xr.Dataset: The updated simulated dataset.

        Raises:
            SimulationError: If multiplier is negative (propagated
                from SimulationEngine).
            DatasetSchemaError: If "RAINFALL" or "TIME" is missing
                (propagated from SimulationEngine).
        """
        return self.simulation.heavy_rainfall(
            start_date,
            end_date,
            multiplier
        )

    def reset_simulation(self) -> None:
        """
        Discard all simulated changes and restore the simulated
        dataset back to the original, unmodified data.

        Returns:
            None.
        """
        self.simulation.reset()
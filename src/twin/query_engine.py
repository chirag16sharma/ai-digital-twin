"""
twin/query_engine.py

Responsible for answering "what's the rainfall at this place on this
date" queries by coordinating SpatialEngine (finds the nearest grid
point) and TemporalEngine (knows the time coordinate name), then
recording the result in StateManager. This is the component that
combines the spatial and temporal dimensions that SpatialEngine and
TemporalEngine each handle independently.
"""

from typing import TypedDict

from src.twin.spatial_engine import SpatialEngine
from src.twin.temporal_engine import TemporalEngine
from src.twin.state_manager import StateManager, DigitalTwinState


class QueryResult(TypedDict):
    """
    Shape of the dictionary returned by QueryEngine.rainfall_query().

    Attributes:
        latitude: The nearest grid latitude actually used for the
            query (may differ slightly from the requested latitude).
        longitude: The nearest grid longitude actually used for the
            query.
        date: The date that was queried, echoed back unchanged.
        rainfall: The rainfall value at that grid point and date, in
            mm, as a plain Python float.
    """
    latitude: float
    longitude: float
    date: str
    rainfall: float


class QueryEngine:
    """
    Answers location-and-date rainfall queries by combining
    SpatialEngine and TemporalEngine, and records each query's result
    in StateManager.

    Single Responsibility:
        Coordination between the spatial and temporal dimensions for
        a single query, plus state recording. This class does not
        implement its own coordinate-finding or rainfall-retrieval
        logic — it delegates entirely to the engines it's given.

    Attributes:
        spatial (SpatialEngine): Used to find the nearest grid point
            and retrieve rainfall for a location.
        temporal (TemporalEngine): Used only for its time_name
            attribute, to select a specific date from a rainfall
            time series.
        state (StateManager): Updated after every successful query,
            and cleared/read via current_state() and reset().
    """

    def __init__(
        self,
        spatial_engine: SpatialEngine,
        temporal_engine: TemporalEngine,
        state_manager: StateManager
    ) -> None:
        """
        Initialize the Query Engine with its three collaborators.

        Args:
            spatial_engine: A SpatialEngine instance, already
                constructed with the target dataset.
            temporal_engine: A TemporalEngine instance, already
                constructed with the same dataset.
            state_manager: A StateManager instance to record query
                results into.
        """
        self.spatial: SpatialEngine = spatial_engine
        self.temporal: TemporalEngine = temporal_engine
        self.state: StateManager = state_manager

    def rainfall_query(
        self,
        latitude: float,
        longitude: float,
        date: str
    ) -> QueryResult:
        """
        Retrieve rainfall for the nearest grid point to a requested
        location, on a specific date, and record the result in state.

        Args:
            latitude: Requested latitude.
            longitude: Requested longitude.
            date: Date string parseable by TemporalEngine's TIME
                coordinate, e.g. "2025-07-15".

        Returns:
            QueryResult: A dict with the resolved grid latitude and
                longitude, the queried date, and the rainfall value
                in mm.

        Raises:
            KeyError: If "RAINFALL" is missing, or if `date` does not
                exist in the dataset's TIME coordinate (propagated
                from the underlying .sel() call).
        """
        # Find nearest grid
        grid_lat, grid_lon = self.spatial.nearest_grid(
            latitude,
            longitude
        )

        # Rainfall time-series
        rainfall_series = self.spatial.rainfall_at(
            latitude,
            longitude
        )

        # Rainfall on requested date
        rainfall: float = rainfall_series.sel(
            {
                self.temporal.time_name: date
            }
        ).values.item()

        # Update state
        self.state.update_state(
            latitude=grid_lat,
            longitude=grid_lon,
            date=date,
            rainfall=rainfall
        )

        return {
            "latitude": grid_lat,
            "longitude": grid_lon,
            "date": date,
            "rainfall": rainfall
        }

    def current_state(self) -> DigitalTwinState:
        """
        Return the Digital Twin's current state, as tracked by
        StateManager.

        Returns:
            DigitalTwinState: The current state dict (latitude,
                longitude, date, rainfall, last_updated).
        """
        return self.state.get_state()

    def reset(self) -> None:
        """
        Clear the Digital Twin's state back to its initial values.

        Returns:
            None.
        """
        self.state.clear_state()
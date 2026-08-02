class QueryEngine:
    """
    Query Engine for the AI Digital Twin.

    Responsibilities
    ----------------
    - Connect Spatial Engine.
    - Connect Temporal Engine.
    - Update State Manager.
    - Return user-friendly answers.
    """

    def __init__(
        self,
        spatial_engine,
        temporal_engine,
        state_manager
    ):
        """
        Initialize Query Engine.
        """

        self.spatial = spatial_engine
        self.temporal = temporal_engine
        self.state = state_manager

    def rainfall_query(
        self,
        latitude,
        longitude,
        date
    ):
        """
        Retrieve rainfall for a location and date.
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
        rainfall = rainfall_series.sel(
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

    def current_state(self):
        """
        Return current Digital Twin state.
        """

        return self.state.get_state()

    def reset(self):
        """
        Clear Digital Twin state.
        """

        self.state.clear_state()
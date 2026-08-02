from pathlib import Path

from src.ingestion.imd_loader import IMDLoader

from src.twin.spatial_engine import SpatialEngine
from src.twin.temporal_engine import TemporalEngine
from src.twin.state_manager import StateManager
from src.twin.query_engine import QueryEngine
from src.twin.simulation_engine import SimulationEngine


class DigitalTwin:
    """
    AI Digital Twin

    Main interface for the complete system.
    """

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        loader = IMDLoader(self.dataset_path)

        self.dataset = loader.load()

        self.spatial = SpatialEngine(self.dataset)

        self.temporal = TemporalEngine(self.dataset)

        self.state = StateManager()

        self.query = QueryEngine(
            self.spatial,
            self.temporal,
            self.state
        )

        self.simulation = SimulationEngine(
            self.dataset
        )

    def rainfall(
        self,
        latitude,
        longitude,
        date
    ):
        """
        Query rainfall.
        """

        return self.query.rainfall_query(
            latitude,
            longitude,
            date
        )

    def current_state(self):
        """
        Return current state.
        """

        return self.query.current_state()

    def reset_state(self):
        """
        Reset state.
        """

        self.query.reset()

    def simulate_increase(self, percentage):
        """
        Increase rainfall.
        """

        return self.simulation.rainfall_increase(
            percentage
        )

    def simulate_decrease(self, percentage):

        return self.simulation.rainfall_decrease(
            percentage
        )

    def simulate_dry_spell(
        self,
        start_date,
        end_date
    ):

        return self.simulation.dry_spell(
            start_date,
            end_date
        )

    def simulate_heavy_rainfall(
        self,
        start_date,
        end_date,
        multiplier
    ):

        return self.simulation.heavy_rainfall(
            start_date,
            end_date,
            multiplier
        )

    def reset_simulation(self):

        self.simulation.reset()
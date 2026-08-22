"""
tests/test_digital_twin.py

Integration tests for twin/digital_twin.py — the top-level class that
wires together IMDLoader, SpatialEngine, TemporalEngine, StateManager,
QueryEngine, and SimulationEngine. These tests exercise the full
construction chain and the public interface end to end, using a
synthetic dataset written to a temporary NetCDF file (so no real IMD
data file is required to run the test suite).
"""

from pathlib import Path

import pytest

from src.exceptions import (
    CoordinateNotFoundError,
    DatasetNotFoundError,
    InvalidCoordinateError,
    InvalidDateError,
    SimulationError,
)
from src.twin.digital_twin import DigitalTwin


@pytest.fixture
def dataset_path(tmp_path: Path, synthetic_rainfall_dataset) -> Path:
    """
    Write the standard synthetic dataset to a temporary .nc file and
    return its path, so DigitalTwin (which loads via IMDLoader from a
    file path, not an in-memory Dataset) can be tested without
    depending on the real IMD dataset existing on disk.

    tmp_path is a built-in pytest fixture: a unique temporary
    directory automatically created and cleaned up per test, so tests
    never collide or leave files behind.
    """
    path = tmp_path / "synthetic_rainfall.nc"
    synthetic_rainfall_dataset.to_netcdf(path)
    return path


@pytest.fixture
def twin(dataset_path: Path) -> DigitalTwin:
    """A fully constructed DigitalTwin over the synthetic dataset."""
    return DigitalTwin(dataset_path)


class TestDigitalTwinConstruction:
    """Tests for DigitalTwin.__init__ and component wiring."""

    def test_constructs_all_components(self, twin):
        """
        Constructing a DigitalTwin should build all five internal
        components, each of the expected type.
        """
        assert twin.spatial is not None
        assert twin.temporal is not None
        assert twin.state is not None
        assert twin.query is not None
        assert twin.simulation is not None

    def test_simulation_engine_time_name_matches_temporal_engine(self, twin):
        """
        This is the Day 4 fix, verified directly: SimulationEngine
        should receive the SAME time_name TemporalEngine detected,
        not independently re-detect it. Proves the two engines can
        never silently disagree on the time coordinate name.
        """
        assert twin.simulation.time_name == twin.temporal.time_name

    def test_raises_dataset_not_found_error_for_missing_file(self, tmp_path):
        """
        Constructing a DigitalTwin with a path that doesn't exist
        should raise DatasetNotFoundError, propagated from IMDLoader.
        """
        missing_path = tmp_path / "does_not_exist.nc"

        with pytest.raises(DatasetNotFoundError):
            DigitalTwin(missing_path)

    def test_raises_coordinate_not_found_error_for_malformed_dataset(
        self, tmp_path
    ):
        """
        Constructing a DigitalTwin from a dataset with no recognizable
        latitude/longitude/time coordinates should raise
        CoordinateNotFoundError, propagated from SpatialEngine
        construction.
        """
        import numpy as np
        import xarray as xr

        malformed = xr.Dataset(
            data_vars={"RAINFALL": (["x", "y"], np.zeros((2, 2)))}
        )
        path = tmp_path / "malformed.nc"
        malformed.to_netcdf(path)

        with pytest.raises(CoordinateNotFoundError):
            DigitalTwin(path)


class TestRainfallDelegation:
    """
    Tests that DigitalTwin.rainfall() correctly delegates to
    QueryEngine and returns the same result shape.
    """

    def test_returns_correct_rainfall_value(self, twin):
        """
        Same known value as the QueryEngine tests: (10.0, 75.0) on
        2025-07-03 (day_idx=2, lat_idx=0, lon_idx=0) = 20.0.
        """
        result = twin.rainfall(10.0, 75.0, "2025-07-03")

        assert result["rainfall"] == pytest.approx(20.0)

    def test_updates_current_state(self, twin):
        """After a query, current_state() should reflect it."""
        twin.rainfall(10.0, 75.0, "2025-07-03")

        state = twin.current_state()

        assert state["date"] == "2025-07-03"
        assert state["rainfall"] == pytest.approx(20.0)

    def test_reset_state_clears_current_state(self, twin):
        """reset_state() should clear state back to all-None."""
        twin.rainfall(10.0, 75.0, "2025-07-03")

        twin.reset_state()

        state = twin.current_state()
        assert state["date"] is None
        assert state["rainfall"] is None

    def test_raises_invalid_coordinate_error_for_bad_location(self, twin):
        """Out-of-range coordinates should propagate as InvalidCoordinateError."""
        with pytest.raises(InvalidCoordinateError):
            twin.rainfall(999.0, 75.0, "2025-07-03")

    def test_raises_invalid_date_error_for_bad_date(self, twin):
        """An unknown date should propagate as InvalidDateError."""
        with pytest.raises(InvalidDateError):
            twin.rainfall(10.0, 75.0, "2099-01-01")


class TestSimulationDelegation:
    """
    Tests that DigitalTwin's simulate_*() methods correctly delegate
    to SimulationEngine.
    """

    def test_simulate_increase_applies_correctly(self, twin):
        """
        Base value at day_idx=1, lat_idx=0, lon_idx=0 is 10.0; a 50%
        increase should produce 15.0.
        """
        result = twin.simulate_increase(50)

        assert float(result["RAINFALL"].values[1, 0, 0]) == pytest.approx(15.0)

    def test_simulate_decrease_raises_for_invalid_percentage(self, twin):
        """A decrease above 100% should raise SimulationError, propagated."""
        with pytest.raises(SimulationError):
            twin.simulate_decrease(150)

    def test_simulate_dry_spell_zeros_target_range(self, twin):
        """dry_spell should zero rainfall within the specified range."""
        result = twin.simulate_dry_spell("2025-07-01", "2025-07-02")

        affected = result["RAINFALL"].sel(TIME=slice("2025-07-01", "2025-07-02"))
        assert float(affected.values.max()) == pytest.approx(0.0)

    def test_simulate_heavy_rainfall_applies_multiplier(self, twin):
        """
        Base value at day_idx=1, lat_idx=0, lon_idx=0 is 10.0;
        multiplier=2.0 should produce 20.0.
        """
        result = twin.simulate_heavy_rainfall(
            "2025-07-02", "2025-07-02", multiplier=2.0
        )

        assert float(result["RAINFALL"].values[1, 0, 0]) == pytest.approx(20.0)

    def test_reset_simulation_restores_original_values(self, twin):
        """After simulate + reset, simulated data should match original."""
        original = twin.dataset["RAINFALL"].values.copy()

        twin.simulate_increase(50)
        twin.reset_simulation()

        import numpy as np
        np.testing.assert_array_almost_equal(
            twin.simulation.simulated_dataset["RAINFALL"].values, original
        )


class TestQueryThenSimulateIndependence:
    """
    Tests that querying and simulating don't interfere with each
    other — QueryEngine reads from twin.dataset (the real, unmodified
    data), while SimulationEngine operates on its own separate
    simulated_dataset. A bug that accidentally pointed both at the
    same object would be a serious, silent correctness issue.
    """

    def test_simulation_does_not_affect_subsequent_queries(self, twin):
        """
        After running a simulation that changes rainfall values,
        a normal rainfall() query should still return the ORIGINAL,
        real value — not the simulated one. This is the core promise
        of keeping SimulationEngine's working copy separate from the
        dataset QueryEngine reads from.
        """
        twin.simulate_increase(1000)  # drastically inflate simulated data

        result = twin.rainfall(10.0, 75.0, "2025-07-03")

        # Real value at day_idx=2, lat_idx=0, lon_idx=0 is 20.0,
        # NOT inflated by the simulation above.
        assert result["rainfall"] == pytest.approx(20.0)
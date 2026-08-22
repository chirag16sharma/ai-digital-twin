"""
tests/test_query_engine.py

Tests for twin/query_engine.py — the component that combines
SpatialEngine (where), TemporalEngine (when), and StateManager
(remembers) into a single rainfall_query() call.
"""

import pytest

from src.exceptions import InvalidCoordinateError, InvalidDateError
from src.twin.query_engine import QueryEngine
from src.twin.spatial_engine import SpatialEngine
from src.twin.state_manager import StateManager
from src.twin.temporal_engine import TemporalEngine


@pytest.fixture
def query_engine(synthetic_rainfall_dataset):
    """
    A fully-wired QueryEngine over the standard synthetic dataset,
    matching how DigitalTwin constructs one — SpatialEngine and
    TemporalEngine built from the same dataset, StateManager fresh.
    """
    spatial = SpatialEngine(synthetic_rainfall_dataset)
    temporal = TemporalEngine(synthetic_rainfall_dataset)
    state = StateManager()

    return QueryEngine(spatial, temporal, state)


class TestRainfallQuery:
    """
    Tests for rainfall_query(), using the fixture's deterministic
    formula: rainfall[day_idx, lat_idx, lon_idx] = (day_idx*10) +
    lat_idx + (lon_idx*0.1).
    """

    def test_returns_correct_rainfall_value(self, query_engine):
        """
        At (latitude=10.0, longitude=75.0) on 2025-07-03 (day_idx=2,
        lat_idx=0, lon_idx=0), expected rainfall is 2*10+0+0 = 20.0.
        """
        result = query_engine.rainfall_query(10.0, 75.0, "2025-07-03")

        assert result["rainfall"] == pytest.approx(20.0)

    def test_returns_grid_snapped_coordinates_not_raw_input(self, query_engine):
        """
        Requesting (10.2, 75.4) should snap to the nearest grid point
        (10.25, 75.5) — the returned latitude/longitude should be the
        SNAPPED values, not an echo of the raw requested values. This
        is the exact distinction flagged since Day 2 as something a
        developer needs to understand, not assume away.
        """
        result = query_engine.rainfall_query(10.2, 75.4, "2025-07-01")

        assert result["latitude"] == 10.25
        assert result["longitude"] == 75.5
        # Confirm it's NOT just echoing the raw input:
        assert result["latitude"] != 10.2
        assert result["longitude"] != 75.4

    def test_echoes_requested_date_unchanged(self, query_engine):
        """
        Unlike latitude/longitude (which get grid-snapped), the date
        should be echoed back exactly as requested — TemporalEngine's
        TIME coordinate uses exact date matching, not nearest-match.
        """
        result = query_engine.rainfall_query(10.0, 75.0, "2025-07-03")

        assert result["date"] == "2025-07-03"

    def test_rainfall_value_is_plain_float(self, query_engine):
        """
        The returned rainfall value should be a native Python float,
        not a numpy scalar or xr.DataArray — confirming the
        .values.item() conversion inside rainfall_query() actually
        does its job. This matters because StateManager's rainfall
        field type depends on this conversion happening correctly.
        """
        result = query_engine.rainfall_query(10.0, 75.0, "2025-07-01")

        assert isinstance(result["rainfall"], float)

    def test_raises_invalid_coordinate_error_for_out_of_range_location(
        self, query_engine
    ):
        """
        An out-of-range latitude should raise InvalidCoordinateError,
        propagated from SpatialEngine — QueryEngine should not
        swallow or re-wrap this.
        """
        with pytest.raises(InvalidCoordinateError):
            query_engine.rainfall_query(999.0, 75.0, "2025-07-01")

    def test_raises_invalid_date_error_for_unknown_date(self, query_engine):
        """
        A date outside the dataset's range should raise
        InvalidDateError — this is QueryEngine's own responsibility
        (not delegated to SpatialEngine or TemporalEngine), since the
        date lookup happens directly inside rainfall_query().
        """
        with pytest.raises(InvalidDateError):
            query_engine.rainfall_query(10.0, 75.0, "2099-01-01")


class TestStateRecording:
    """Tests that rainfall_query() correctly records results into StateManager."""

    def test_successful_query_updates_state(self, query_engine):
        """After a successful query, current_state() should reflect it."""
        query_engine.rainfall_query(10.0, 75.0, "2025-07-03")

        state = query_engine.current_state()

        assert state["latitude"] == 10.0
        assert state["longitude"] == 75.0
        assert state["date"] == "2025-07-03"
        assert state["rainfall"] == pytest.approx(20.0)

    def test_state_reflects_grid_snapped_coordinates(self, query_engine):
        """
        State should store the SNAPPED coordinates, matching what
        rainfall_query() returns — not the raw requested values.
        """
        query_engine.rainfall_query(10.2, 75.4, "2025-07-01")

        state = query_engine.current_state()

        assert state["latitude"] == 10.25
        assert state["longitude"] == 75.5

    def test_failed_query_does_not_update_state(self, query_engine):
        """
        If rainfall_query() raises (bad date), state should remain
        untouched from before the failed call — the failure happens
        before state.update_state() is ever reached.
        """
        query_engine.rainfall_query(10.0, 75.0, "2025-07-01")
        state_before = dict(query_engine.current_state())

        with pytest.raises(InvalidDateError):
            query_engine.rainfall_query(10.0, 75.0, "2099-01-01")

        state_after = query_engine.current_state()

        assert state_after["date"] == state_before["date"]
        assert state_after["rainfall"] == state_before["rainfall"]

    def test_second_query_overwrites_first(self, query_engine):
        """A second successful query should overwrite the first query's state."""
        query_engine.rainfall_query(10.0, 75.0, "2025-07-01")
        query_engine.rainfall_query(10.5, 75.5, "2025-07-05")

        state = query_engine.current_state()

        assert state["latitude"] == 10.5
        assert state["longitude"] == 75.5
        assert state["date"] == "2025-07-05"


class TestReset:
    """Tests for reset()."""

    def test_reset_clears_state_after_query(self, query_engine):
        """After a query, reset() should clear state back to all-None fields."""
        query_engine.rainfall_query(10.0, 75.0, "2025-07-01")

        query_engine.reset()

        state = query_engine.current_state()
        assert state["latitude"] is None
        assert state["longitude"] is None
        assert state["date"] is None
        assert state["rainfall"] is None
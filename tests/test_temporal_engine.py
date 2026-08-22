"""
tests/test_temporal_engine.py

Tests for twin/temporal_engine.py — time coordinate detection, date
range reporting, and date/date-range-based rainfall retrieval across
the full spatial grid.
"""

import numpy as np
import pytest
import xarray as xr

from src.exceptions import CoordinateNotFoundError, DatasetSchemaError, InvalidDateError
from src.twin.temporal_engine import TemporalEngine


class TestTemporalEngineConstruction:
    """Tests for TemporalEngine.__init__ and time coordinate detection."""

    def test_detects_uppercase_time_coordinate(self, synthetic_rainfall_dataset):
        """On the standard fixture, TemporalEngine should detect 'TIME'."""
        engine = TemporalEngine(synthetic_rainfall_dataset)

        assert engine.time_name == "TIME"

    def test_detects_lowercase_time_coordinate(self, synthetic_dataset_lowercase_coords):
        """On the lowercase fixture, TemporalEngine should detect 'time'."""
        engine = TemporalEngine(synthetic_dataset_lowercase_coords)

        assert engine.time_name == "time"

    def test_caches_times_array_with_correct_length(self, synthetic_rainfall_dataset):
        """
        The fixture has 5 days (2025-07-01 through 2025-07-05), so
        self.times should have exactly 5 entries.
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        assert len(engine.times) == 5

    def test_raises_when_no_time_coordinate_found(self):
        """
        Constructing a TemporalEngine on a dataset with no
        recognizable time coordinate should raise
        CoordinateNotFoundError.
        """
        dataset = xr.Dataset(
            coords={
                "LATITUDE": np.array([10.0]),
                "LONGITUDE": np.array([75.0]),
            }
        )

        with pytest.raises(CoordinateNotFoundError):
            TemporalEngine(dataset)


class TestDateRangeReporting:
    """Tests for first_date(), last_date(), and number_of_days()."""

    def test_first_date_is_earliest_timestamp(self, synthetic_rainfall_dataset):
        """first_date() should return 2025-07-01, the fixture's earliest date."""
        engine = TemporalEngine(synthetic_rainfall_dataset)

        assert engine.first_date() == np.datetime64("2025-07-01")

    def test_last_date_is_latest_timestamp(self, synthetic_rainfall_dataset):
        """last_date() should return 2025-07-05, the fixture's latest date."""
        engine = TemporalEngine(synthetic_rainfall_dataset)

        assert engine.last_date() == np.datetime64("2025-07-05")

    def test_number_of_days_matches_fixture(self, synthetic_rainfall_dataset):
        """The fixture spans exactly 5 days."""
        engine = TemporalEngine(synthetic_rainfall_dataset)

        assert engine.number_of_days() == 5


class TestGetDate:
    """
    Tests for get_date(), using the fixture's deterministic formula:
    rainfall[day_idx, lat_idx, lon_idx] = (day_idx*10) + lat_idx +
    (lon_idx*0.1). get_date() returns the full 3x3 spatial grid for
    one date, unlike SpatialEngine.rainfall_at() which returns a
    single point's full time series.
    """

    def test_returns_correct_grid_for_known_date(self, synthetic_rainfall_dataset):
        """
        2025-07-03 is day_idx=2, so the expected 3x3 grid is:
            [[20.0, 20.1, 20.2],
             [21.0, 21.1, 21.2],
             [22.0, 22.1, 22.2]]
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        result = engine.get_date("2025-07-03")

        expected = np.array([
            [20.0, 20.1, 20.2],
            [21.0, 21.1, 21.2],
            [22.0, 22.1, 22.2],
        ])
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_returns_correct_grid_for_first_date(self, synthetic_rainfall_dataset):
        """
        2025-07-01 is day_idx=0, so the expected grid is all base
        values with no day offset:
            [[0.0, 0.1, 0.2],
             [1.0, 1.1, 1.2],
             [2.0, 2.1, 2.2]]
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        result = engine.get_date("2025-07-01")

        expected = np.array([
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
        ])
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_raises_invalid_date_error_for_unknown_date(
        self, synthetic_rainfall_dataset
    ):
        """
        A date outside the fixture's range (2025-07-01 through
        2025-07-05) should raise InvalidDateError — a bad-input
        problem, distinct from a schema problem.
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        with pytest.raises(InvalidDateError):
            engine.get_date("2099-01-01")

    def test_invalid_date_error_message_includes_available_range(
        self, synthetic_rainfall_dataset
    ):
        """
        The raised error's message should mention the dataset's
        actual available date range, so a developer immediately
        knows what dates ARE valid, not just that this one wasn't.
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        with pytest.raises(InvalidDateError) as exc_info:
            engine.get_date("2099-01-01")

        message = str(exc_info.value)
        assert "2025-07-01" in message
        assert "2025-07-05" in message

    def test_raises_dataset_schema_error_when_rainfall_variable_missing(self):
        """
        If the dataset has a valid time coordinate but no RAINFALL
        variable, get_date() should raise DatasetSchemaError — a
        genuinely different failure than an unknown date.
        """
        dataset = xr.Dataset(
            coords={
                "TIME": np.array(
                    ["2025-07-01", "2025-07-02"], dtype="datetime64[ns]"
                ),
            }
        )
        engine = TemporalEngine(dataset)

        with pytest.raises(DatasetSchemaError):
            engine.get_date("2025-07-01")


class TestGetDateRange:
    """Tests for get_date_range()."""

    def test_returns_correct_number_of_days_in_range(self, synthetic_rainfall_dataset):
        """
        2025-07-02 through 2025-07-04 (inclusive) should return 3
        days: day_idx 1, 2, 3.
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        result = engine.get_date_range("2025-07-02", "2025-07-04")

        assert result.sizes["TIME"] == 3

    def test_range_is_inclusive_of_both_endpoints(self, synthetic_rainfall_dataset):
        """
        Requesting the full fixture range (2025-07-01 to 2025-07-05)
        should return all 5 days, confirming both endpoints are
        included, not just the start.
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        result = engine.get_date_range("2025-07-01", "2025-07-05")

        assert result.sizes["TIME"] == 5

    def test_first_grid_point_values_match_expected_days(
        self, synthetic_rainfall_dataset
    ):
        """
        At grid point (lat_idx=0, lon_idx=0), the range 2025-07-02
        through 2025-07-04 should return values [10.0, 20.0, 30.0]
        (day_idx 1, 2, 3 — each day_idx*10 at lat_idx=0, lon_idx=0).
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        result = engine.get_date_range("2025-07-02", "2025-07-04")
        point_series = result.isel(LATITUDE=0, LONGITUDE=0)

        np.testing.assert_array_almost_equal(
            point_series.values, [10.0, 20.0, 30.0]
        )

    def test_non_overlapping_range_returns_empty_result_without_raising(
        self, synthetic_rainfall_dataset
    ):
        """
        A date range entirely outside the fixture's coverage should
        NOT raise — per the documented design decision in
        TemporalEngine.get_date_range(), an empty result is returned
        (and a warning logged) rather than treated as fatal, since a
        caller might legitimately want to handle 'no data here'
        gracefully rather than via exception handling.
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        result = engine.get_date_range("2099-01-01", "2099-01-05")

        assert result.sizes["TIME"] == 0
        
    class TestAvailableDates:
        """Tests for available_dates() — console output only."""

    def test_prints_date_range_and_day_count(
        self, synthetic_rainfall_dataset, capsys
    ):
        """
        No test previously called this method (confirmed via the
        Day 5 coverage report) — it prints first_date(), last_date(),
        and number_of_days().
        """
        engine = TemporalEngine(synthetic_rainfall_dataset)

        engine.available_dates()

        captured = capsys.readouterr()
        assert "2025-07-01" in captured.out
        assert "2025-07-05" in captured.out
        assert "Total Days : 5" in captured.out


class TestGetDateRangeSchemaError:
    """
    get_date() already had a DatasetSchemaError test for a missing
    RAINFALL variable; get_date_range() did not (confirmed via the
    Day 5 coverage report) — closing that gap here.
    """

    def test_raises_dataset_schema_error_when_rainfall_missing(self):
        """A dataset with a TIME coordinate but no RAINFALL variable."""
        dataset = xr.Dataset(
            coords={
                "TIME": np.array(
                    ["2025-07-01", "2025-07-02"], dtype="datetime64[ns]"
                ),
            }
        )
        engine = TemporalEngine(dataset)

        with pytest.raises(DatasetSchemaError):
            engine.get_date_range("2025-07-01", "2025-07-02")
"""
tests/test_spatial_engine.py

Tests for twin/spatial_engine.py — coordinate detection (via
find_coordinate), nearest-grid-point lookup, coordinate bounds
validation, and location-based rainfall retrieval.
"""

import numpy as np
import pytest
import xarray as xr

from src.exceptions import CoordinateNotFoundError, DatasetSchemaError, InvalidCoordinateError, InvalidDateError
from src.twin.spatial_engine import SpatialEngine   
class TestSpatialEngineConstruction:
    """Tests for SpatialEngine.__init__ and coordinate auto-detection."""

    def test_detects_uppercase_coordinate_names(self, synthetic_rainfall_dataset):
        """
        On the standard fixture (uppercase TIME/LATITUDE/LONGITUDE),
        SpatialEngine should detect and store the exact names found.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        assert engine.lat_name == "LATITUDE"
        assert engine.lon_name == "LONGITUDE"
        assert engine.time_name == "TIME"

    def test_detects_lowercase_coordinate_names(self, synthetic_dataset_lowercase_coords):
        """
        On the lowercase fixture, SpatialEngine should still construct
        successfully and detect the lowercase names — proving it
        doesn't assume a fixed casing.
        """
        engine = SpatialEngine(synthetic_dataset_lowercase_coords)

        assert engine.lat_name == "latitude"
        assert engine.lon_name == "longitude"
        assert engine.time_name == "time"

    def test_caches_latitude_and_longitude_arrays(self, synthetic_rainfall_dataset):
        """
        latitudes/longitudes should be cached as numpy arrays matching
        the dataset's actual coordinate values, so nearest-neighbor
        lookups don't need to re-read the dataset every call.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        np.testing.assert_array_equal(
            engine.latitudes, np.array([10.0, 10.25, 10.5])
        )
        np.testing.assert_array_equal(
            engine.longitudes, np.array([75.0, 75.25, 75.5])
        )

    def test_raises_when_no_latitude_coordinate_found(
        self, synthetic_dataset_missing_latitude
    ):
        """
        Constructing a SpatialEngine on a dataset with no recognizable
        latitude coordinate should raise CoordinateNotFoundError,
        propagated from find_coordinate() — not fail some other,
        less specific way.
        """
        with pytest.raises(CoordinateNotFoundError):
            SpatialEngine(synthetic_dataset_missing_latitude)


class TestNearestLatitudeLongitude:
    """Tests for nearest_latitude() and nearest_longitude()."""

    def test_exact_match_returns_same_value(self, synthetic_rainfall_dataset):
        """Requesting an exact grid latitude should return that same value."""
        engine = SpatialEngine(synthetic_rainfall_dataset)

        assert engine.nearest_latitude(10.25) == 10.25

    def test_nearest_latitude_rounds_to_closer_grid_point(
        self, synthetic_rainfall_dataset
    ):
        """
        Grid latitudes are [10.0, 10.25, 10.5]. Requesting 10.2 is
        closer to 10.25 (distance 0.05) than to 10.0 (distance 0.2),
        so 10.25 should be returned.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        assert engine.nearest_latitude(10.2) == 10.25

    def test_nearest_longitude_rounds_to_closer_grid_point(
        self, synthetic_rainfall_dataset
    ):
        """
        Grid longitudes are [75.0, 75.25, 75.5]. Requesting 75.4 is
        closer to 75.5 (distance 0.1) than to 75.25 (distance 0.15),
        so 75.5 should be returned.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        assert engine.nearest_longitude(75.4) == 75.5

    def test_nearest_latitude_raises_when_out_of_range(
        self, synthetic_rainfall_dataset
    ):
        """
        Requesting a latitude far outside the dataset's coverage
        ([10.0, 10.5]) should raise InvalidCoordinateError, not
        silently snap to the nearest edge value.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        with pytest.raises(InvalidCoordinateError):
            engine.nearest_latitude(999.0)

    def test_nearest_longitude_raises_when_out_of_range(
        self, synthetic_rainfall_dataset
    ):
        """
        Same as above, for longitude — coverage is [75.0, 75.5].
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        with pytest.raises(InvalidCoordinateError):
            engine.nearest_longitude(-40.0)

    def test_boundary_values_are_accepted(self, synthetic_rainfall_dataset):
        """
        The exact min/max boundary values should be accepted (not
        rejected as 'out of range') — validation uses <= on both
        sides, so the edges of the range are valid, not exclusive.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        assert engine.nearest_latitude(10.0) == 10.0
        assert engine.nearest_latitude(10.5) == 10.5


class TestNearestGrid:
    """Tests for nearest_grid(), which combines lat + lon lookup."""

    def test_returns_nearest_lat_lon_pair(self, synthetic_rainfall_dataset):
        """nearest_grid() should return the combined nearest (lat, lon) pair."""
        engine = SpatialEngine(synthetic_rainfall_dataset)

        lat, lon = engine.nearest_grid(10.2, 75.4)

        assert lat == 10.25
        assert lon == 75.5

    def test_raises_if_either_coordinate_is_out_of_range(
        self, synthetic_rainfall_dataset
    ):
        """
        If latitude is valid but longitude is out of range (or vice
        versa), nearest_grid() should still raise — the whole request
        is invalid if any part of it is.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        with pytest.raises(InvalidCoordinateError):
            engine.nearest_grid(10.25, 999.0)


class TestRainfallAt:
    """
    Tests for rainfall_at(), using the fixture's deterministic
    formula: rainfall[day, lat_idx, lon_idx] = (day*10) + lat_idx +
    (lon_idx * 0.1). At grid point (lat_idx=0, lon_idx=0), the
    expected time series is [0, 10, 20, 30, 40] across the 5 days.
    """

    def test_returns_correct_time_series_at_first_grid_point(
        self, synthetic_rainfall_dataset
    ):
        """
        At (latitude=10.0, longitude=75.0) — lat_idx=0, lon_idx=0 —
        the expected rainfall series across all 5 days is
        [0, 10, 20, 30, 40], per the fixture's formula.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_at(10.0, 75.0)

        np.testing.assert_array_almost_equal(
            result.values, [0.0, 10.0, 20.0, 30.0, 40.0]
        )

    def test_returns_correct_time_series_at_nonzero_grid_point(
        self, synthetic_rainfall_dataset
    ):
        """
        At (latitude=10.25, longitude=75.5) — lat_idx=1, lon_idx=2 —
        the expected rainfall series is [1.2, 11.2, 21.2, 31.2, 41.2],
        per the fixture's formula: day*10 + 1 + (2*0.1).
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_at(10.25, 75.5)

        np.testing.assert_array_almost_equal(
            result.values, [1.2, 11.2, 21.2, 31.2, 41.2]
        )

    def test_snaps_nearby_request_to_correct_grid_point(
        self, synthetic_rainfall_dataset
    ):
        """
        A request that doesn't land exactly on a grid point (10.2,
        75.4) should snap to (10.25, 75.5) and return that grid
        point's series — not interpolate, not error.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_at(10.2, 75.4)

        np.testing.assert_array_almost_equal(
            result.values, [1.2, 11.2, 21.2, 31.2, 41.2]
        )

    def test_raises_dataset_schema_error_when_rainfall_variable_missing(self):
        """
        If the dataset has valid coordinates but no RAINFALL variable
        at all, rainfall_at() should raise DatasetSchemaError rather
        than a raw KeyError.
        """
        dataset = xr.Dataset(
            coords={
                "TIME": np.array(["2025-07-01"], dtype="datetime64[ns]"),
                "LATITUDE": np.array([10.0]),
                "LONGITUDE": np.array([75.0]),
            }
        )
        engine = SpatialEngine(dataset)

        with pytest.raises(DatasetSchemaError):
            engine.rainfall_at(10.0, 75.0)


class TestRainfallOnDate:
    """Tests for rainfall_on_date(), combining spatial + a single date."""

    def test_returns_correct_single_value(self, synthetic_rainfall_dataset):
        """
        At (lat_idx=0, lon_idx=0) on day index 2 (2025-07-03), the
        expected value is 2*10 + 0 + 0 = 20.0.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_on_date(10.0, 75.0, "2025-07-03")

        assert float(result.values) == pytest.approx(20.0)

    def test_raises_invalid_date_error_for_unknown_date(
        self, synthetic_rainfall_dataset
    ):
        """
        Requesting a date outside the fixture's range (2025-07-01
        through 2025-07-05) should raise InvalidDateError — this now
        matches TemporalEngine.get_date()'s behavior for the same
        conceptual failure, after the Day 5 fix to
        SpatialEngine.rainfall_on_date().
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        with pytest.raises(InvalidDateError):
            engine.rainfall_on_date(10.0, 75.0, "2099-01-01")
            
    class TestAvailableCoordinates:

        """Tests for available_coordinates() — console output only."""

    def test_prints_latitude_and_longitude_ranges(
        self, synthetic_rainfall_dataset, capsys
    ):
        """
        No test previously called this method at all (confirmed via
        the Day 5 coverage report) — it prints the min/max of both
        latitude and longitude.
        """
        engine = SpatialEngine(synthetic_rainfall_dataset)

        engine.available_coordinates()

        captured = capsys.readouterr()
        assert "Latitude" in captured.out
        assert "10.0" in captured.out
        assert "10.5" in captured.out
        assert "Longitude" in captured.out
        assert "75.0" in captured.out
        assert "75.5" in captured.out
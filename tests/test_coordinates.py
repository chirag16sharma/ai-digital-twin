"""
tests/test_coordinates.py

Tests for src/utils/coordinates.py — the shared coordinate-detection
logic used by SpatialEngine, TemporalEngine, and (as a fallback) by
SimulationEngine. This is the lowest-level piece of the twin/ package
that has its own dedicated test file, since every engine's
construction depends on it working correctly.
"""

import pytest

from config.settings import LATITUDE_ALIASES, LONGITUDE_ALIASES, TIME_ALIASES
from src.exceptions import CoordinateNotFoundError
from src.utils.coordinates import find_coordinate


class TestFindCoordinateUppercase:
    """Tests find_coordinate() against the standard uppercase-named fixture."""

    def test_finds_uppercase_latitude(self, synthetic_rainfall_dataset):
        """LATITUDE should be found when the dataset uses uppercase naming."""
        result = find_coordinate(synthetic_rainfall_dataset, LATITUDE_ALIASES)

        assert result == "LATITUDE"

    def test_finds_uppercase_longitude(self, synthetic_rainfall_dataset):
        """LONGITUDE should be found when the dataset uses uppercase naming."""
        result = find_coordinate(synthetic_rainfall_dataset, LONGITUDE_ALIASES)

        assert result == "LONGITUDE"

    def test_finds_uppercase_time(self, synthetic_rainfall_dataset):
        """TIME should be found when the dataset uses uppercase naming."""
        result = find_coordinate(synthetic_rainfall_dataset, TIME_ALIASES)

        assert result == "TIME"


class TestFindCoordinateLowercase:
    """
    Tests find_coordinate() against the lowercase-named fixture — this
    is the whole reason find_coordinate() exists in the first place,
    since real IMD files are inconsistent about casing.
    """

    def test_finds_lowercase_latitude(self, synthetic_dataset_lowercase_coords):
        """latitude should be found when the dataset uses lowercase naming."""
        result = find_coordinate(synthetic_dataset_lowercase_coords, LATITUDE_ALIASES)

        assert result == "latitude"

    def test_finds_lowercase_longitude(self, synthetic_dataset_lowercase_coords):
        """longitude should be found when the dataset uses lowercase naming."""
        result = find_coordinate(synthetic_dataset_lowercase_coords, LONGITUDE_ALIASES)

        assert result == "longitude"

    def test_finds_lowercase_time(self, synthetic_dataset_lowercase_coords):
        """time should be found when the dataset uses lowercase naming."""
        result = find_coordinate(synthetic_dataset_lowercase_coords, TIME_ALIASES)

        assert result == "time"


class TestFindCoordinateNotFound:
    """
    Tests that find_coordinate() raises CoordinateNotFoundError (not
    a generic exception, and not a silent wrong answer) when none of
    the candidate names exist in the dataset.
    """

    def test_raises_when_latitude_missing(self, synthetic_dataset_missing_latitude):
        """
        Requesting a latitude coordinate from a dataset that has none
        should raise CoordinateNotFoundError specifically.
        """
        with pytest.raises(CoordinateNotFoundError):
            find_coordinate(synthetic_dataset_missing_latitude, LATITUDE_ALIASES)

    def test_error_message_includes_attempted_names(
        self, synthetic_dataset_missing_latitude
    ):
        """
        The raised error's message should mention the alias list that
        was tried, so a developer reading the error immediately knows
        which names were checked — not just that "something" wasn't
        found.
        """
        with pytest.raises(CoordinateNotFoundError) as exc_info:
            find_coordinate(synthetic_dataset_missing_latitude, LATITUDE_ALIASES)

        assert "LATITUDE" in str(exc_info.value)
        assert "latitude" in str(exc_info.value)
        assert "lat" in str(exc_info.value)

    def test_missing_latitude_dataset_still_has_other_coords(
        self, synthetic_dataset_missing_latitude
    ):
        """
        Sanity check on the fixture itself: TIME and LONGITUDE should
        still be findable on this dataset — only latitude is
        deliberately absent. This confirms the fixture is testing
        exactly one missing coordinate, not accidentally broken in
        some other way.
        """
        assert find_coordinate(synthetic_dataset_missing_latitude, TIME_ALIASES) == "TIME"
        assert (
            find_coordinate(synthetic_dataset_missing_latitude, LONGITUDE_ALIASES)
            == "LONGITUDE"
        )


class TestFindCoordinatePriorityOrder:
    """
    Tests that find_coordinate() respects the priority order of the
    candidate list — returning the FIRST match, not just any match.
    """

    def test_returns_first_match_when_multiple_aliases_could_match(self):
        """
        If a dataset happened to have both 'LATITUDE' and 'latitude'
        as separate coordinates (an edge case, but worth confirming
        behavior), find_coordinate() should return whichever comes
        first in the possible_names list — since that's the
        documented contract ("first match found is returned").
        """
        import numpy as np
        import xarray as xr

        dataset = xr.Dataset(
            coords={
                "LATITUDE": np.array([10.0, 20.0]),
                "latitude": np.array([1.0, 2.0]),
            }
        )

        result = find_coordinate(dataset, ["LATITUDE", "latitude", "lat"])

        assert result == "LATITUDE"
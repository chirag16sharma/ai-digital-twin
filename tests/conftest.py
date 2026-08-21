"""
tests/conftest.py

Shared pytest fixtures for the AI Digital Twin test suite. Builds
small, synthetic, deterministic xarray Datasets so tests never depend
on the real (large) IMD dataset existing on disk. This keeps tests
fast, self-contained, and safe to run in CI without any data files.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _build_rainfall_grid() -> np.ndarray:
    """
    Build a deterministic 5x3x3 rainfall array shared by the clean
    and dirty dataset fixtures below.

    Formula: rainfall[day, lat, lon] = (day * 10) + lat + (lon * 0.1)

    This is deliberately deterministic (not random) so any test can
    compute the exact expected value for any cell without needing to
    inspect the fixture's source — e.g. cell [2, 1, 2] should always
    be (2*10) + 1 + (2*0.1) = 21.2, regardless of who's reading the
    test or when.
    """
    rainfall = np.zeros((5, 3, 3))
    for day_idx in range(5):
        for lat_idx in range(3):
            for lon_idx in range(3):
                rainfall[day_idx, lat_idx, lon_idx] = (
                    (day_idx * 10) + lat_idx + (lon_idx * 0.1)
                )
    return rainfall


@pytest.fixture
def synthetic_rainfall_dataset() -> xr.Dataset:
    """
    A small, clean, deterministic rainfall dataset using the
    standard uppercase coordinate names (TIME, LATITUDE, LONGITUDE)
    and RAINFALL variable.

    Grid:
        Latitudes:  [10.0, 10.25, 10.5]
        Longitudes: [75.0, 75.25, 75.5]
        Dates:      2025-07-01 through 2025-07-05 (5 days)

    Returns:
        xr.Dataset: A clean dataset with no NaN or negative values —
            suitable for SpatialEngine, TemporalEngine, QueryEngine,
            SimulationEngine, and DigitalTwin tests. Not suitable for
            DataCleaner tests — see synthetic_dirty_rainfall_dataset.
    """
    return xr.Dataset(
        data_vars={
            "RAINFALL": (["TIME", "LATITUDE", "LONGITUDE"], _build_rainfall_grid()),
        },
        coords={
            "TIME": pd.date_range("2025-07-01", periods=5, freq="D"),
            "LATITUDE": np.array([10.0, 10.25, 10.5]),
            "LONGITUDE": np.array([75.0, 75.25, 75.5]),
        },
    )


@pytest.fixture
def synthetic_dirty_rainfall_dataset() -> xr.Dataset:
    """
    Same shape/grid as synthetic_rainfall_dataset, but with known
    missing (NaN) and negative rainfall values deliberately injected
    — for testing DataCleaner.quality_report() and clean().

    Known issues, by design:
        - rainfall[0, 0, 0] is NaN (exactly 1 missing value)
        - rainfall[1, 1, 1] is -5.0 (exactly 1 negative value)
        Every other cell matches the same deterministic formula as
        synthetic_rainfall_dataset, so cleaning behavior can be
        checked cell-by-cell, and counts (1 missing, 1 negative) are
        exact, known numbers to assert against.

    Returns:
        xr.Dataset: A dataset with exactly one NaN and one negative
            rainfall value, everything else valid.
    """
    rainfall = _build_rainfall_grid()
    rainfall[0, 0, 0] = np.nan
    rainfall[1, 1, 1] = -5.0

    return xr.Dataset(
        data_vars={
            "RAINFALL": (["TIME", "LATITUDE", "LONGITUDE"], rainfall),
        },
        coords={
            "TIME": pd.date_range("2025-07-01", periods=5, freq="D"),
            "LATITUDE": np.array([10.0, 10.25, 10.5]),
            "LONGITUDE": np.array([75.0, 75.25, 75.5]),
        },
    )


@pytest.fixture
def synthetic_dataset_lowercase_coords() -> xr.Dataset:
    """
    Same grid/values as synthetic_rainfall_dataset, but using
    lowercase coordinate names (time, latitude, longitude) instead of
    uppercase — for testing find_coordinate()'s alias detection
    (config.settings.*_ALIASES), which exists precisely because real
    IMD files vary in coordinate-name casing across versions.

    Returns:
        xr.Dataset: A clean dataset identical in values to
            synthetic_rainfall_dataset, but with lowercase coordinate
            names.
    """
    return xr.Dataset(
        data_vars={
            "RAINFALL": (["time", "latitude", "longitude"], _build_rainfall_grid()),
        },
        coords={
            "time": pd.date_range("2025-07-01", periods=5, freq="D"),
            "latitude": np.array([10.0, 10.25, 10.5]),
            "longitude": np.array([75.0, 75.25, 75.5]),
        },
    )


@pytest.fixture
def synthetic_dataset_missing_latitude() -> xr.Dataset:
    """
    A dataset missing a recognizable latitude coordinate entirely —
    for testing that find_coordinate() (and therefore SpatialEngine
    construction) raises CoordinateNotFoundError as expected, rather
    than failing some other, less specific way.

    Returns:
        xr.Dataset: A dataset with TIME and LONGITUDE, but no
            latitude coordinate under any recognized alias.
    """
    return xr.Dataset(
        data_vars={
            "RAINFALL": (["TIME", "LONGITUDE"], np.zeros((5, 3))),
        },
        coords={
            "TIME": pd.date_range("2025-07-01", periods=5, freq="D"),
            "LONGITUDE": np.array([75.0, 75.25, 75.5]),
        },
    )
"""
tests/test_data_explorer.py

Tests for preprocessing/data_explorer.py — dataset summary reporting.
Since summary() only prints (no return value), these tests rely on
pytest's capsys fixture to capture and assert on console output.
"""

import numpy as np
import pytest
import xarray as xr

from src.exceptions import DatasetSchemaError
from src.preprocessing.data_explorer import DataExplorer


class TestSummary:
    """
    Tests for summary(), using the fixture's known values: 5 time
    steps, 3 latitudes, 3 longitudes, rainfall range determined by
    the deterministic formula (day_idx*10 + lat_idx + lon_idx*0.1).
    Min is 0.0 (day=0,lat=0,lon=0), max is 42.2 (day=4,lat=2,lon=2).
    """

    def test_prints_correct_dimension_counts(self, synthetic_rainfall_dataset, capsys):
        """summary() should report 5 time steps, 3 latitudes, 3 longitudes."""
        explorer = DataExplorer(synthetic_rainfall_dataset)

        explorer.summary()

        captured = capsys.readouterr()
        assert "Time Steps : 5" in captured.out
        assert "Latitudes  : 3" in captured.out
        assert "Longitudes : 3" in captured.out

    def test_prints_correct_rainfall_statistics(self, synthetic_rainfall_dataset, capsys):
        """
        Minimum should be 0.00 mm (day=0,lat=0,lon=0), maximum should
        be 42.20 mm (day=4,lat=2,lon=2), per the fixture's formula.
        """
        explorer = DataExplorer(synthetic_rainfall_dataset)

        explorer.summary()

        captured = capsys.readouterr()
        assert "Minimum : 0.00 mm" in captured.out
        assert "Maximum : 42.20 mm" in captured.out

    def test_prints_date_range(self, synthetic_rainfall_dataset, capsys):
        """summary() should print both the first and last date."""
        explorer = DataExplorer(synthetic_rainfall_dataset)

        explorer.summary()

        captured = capsys.readouterr()
        assert "2025-07-01" in captured.out
        assert "2025-07-05" in captured.out

    def test_works_with_lowercase_coordinate_names(
        self, synthetic_dataset_lowercase_coords, capsys
    ):
        """
        summary() should work identically on the lowercase-named
        fixture — this is the Day 4 fix verified directly: before
        that fix, DataExplorer assumed uppercase TIME/LATITUDE/
        LONGITUDE and would have raised a raw KeyError here instead.
        """
        explorer = DataExplorer(synthetic_dataset_lowercase_coords)

        explorer.summary()

        captured = capsys.readouterr()
        assert "Time Steps : 5" in captured.out
        assert "Latitudes  : 3" in captured.out
        assert "Longitudes : 3" in captured.out

    def test_raises_dataset_schema_error_when_rainfall_missing(self):
        """
        A dataset with valid coordinates but no RAINFALL variable
        should raise DatasetSchemaError, not a raw KeyError.
        """
        dataset = xr.Dataset(
            coords={
                "TIME": np.array(["2025-07-01"], dtype="datetime64[ns]"),
                "LATITUDE": np.array([10.0]),
                "LONGITUDE": np.array([75.0]),
            }
        )
        explorer = DataExplorer(dataset)

        with pytest.raises(DatasetSchemaError):
            explorer.summary()

    def test_raises_dataset_schema_error_when_latitude_missing(
        self, synthetic_dataset_missing_latitude
    ):
        """
        A dataset missing a recognizable latitude coordinate should
        raise DatasetSchemaError (via CoordinateNotFoundError, which
        is a DatasetSchemaError subclass) when summary() tries to
        detect it via find_coordinate().
        """
        explorer = DataExplorer(synthetic_dataset_missing_latitude)

        with pytest.raises(DatasetSchemaError):
            explorer.summary()
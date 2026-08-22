"""
tests/test_imd_loader.py

Tests for src/ingestion/imd_loader.py — the entry point of the data
pipeline. Loading a real (small, synthetic) NetCDF file from a
temporary path, and confirming the missing-file failure mode raises
the correct domain exception.
"""

from pathlib import Path

import pytest

from src.exceptions import DatasetNotFoundError
from src.ingestion.imd_loader import IMDLoader


@pytest.fixture
def dataset_file(tmp_path: Path, synthetic_rainfall_dataset) -> Path:
    """
    Write the standard synthetic dataset to a temporary .nc file,
    for testing IMDLoader.load() against a real file on disk.
    """
    path = tmp_path / "test_rainfall.nc"
    synthetic_rainfall_dataset.to_netcdf(path)
    return path


class TestIMDLoaderConstruction:
    """Tests for IMDLoader.__init__."""

    def test_accepts_string_path(self, dataset_file: Path):
        """__init__ should accept a plain string path, not just a Path object."""
        loader = IMDLoader(str(dataset_file))

        assert loader.file_path == dataset_file

    def test_accepts_path_object(self, dataset_file: Path):
        """__init__ should accept a Path object directly."""
        loader = IMDLoader(dataset_file)

        assert loader.file_path == dataset_file

    def test_does_not_validate_path_at_construction(self, tmp_path: Path):
        """
        Constructing an IMDLoader with a path that doesn't exist yet
        should NOT raise — validation is documented to happen at
        load() time, not construction time, so a loader can be built
        before the file exists (e.g. before a download completes).
        """
        missing_path = tmp_path / "not_yet_downloaded.nc"

        loader = IMDLoader(missing_path)  # should not raise

        assert loader.file_path == missing_path


class TestIMDLoaderLoad:
    """Tests for IMDLoader.load()."""

    def test_loads_dataset_successfully(self, dataset_file: Path):
        """load() should return an xr.Dataset containing RAINFALL."""
        loader = IMDLoader(dataset_file)

        result = loader.load()

        assert "RAINFALL" in result.data_vars

    def test_loaded_dataset_preserves_values(
        self, dataset_file: Path, synthetic_rainfall_dataset
    ):
        """
        The loaded dataset's values should exactly match what was
        written — confirming the disk round-trip (write via
        to_netcdf, read via IMDLoader.load()) doesn't silently
        corrupt or alter data.
        """
        import numpy as np

        loader = IMDLoader(dataset_file)

        result = loader.load()

        np.testing.assert_array_almost_equal(
            result["RAINFALL"].values,
            synthetic_rainfall_dataset["RAINFALL"].values,
        )

    def test_raises_dataset_not_found_error_for_missing_file(self, tmp_path: Path):
        """
        load() on a nonexistent path should raise
        DatasetNotFoundError, not the built-in FileNotFoundError
        (replaced on Day 3).
        """
        missing_path = tmp_path / "does_not_exist.nc"
        loader = IMDLoader(missing_path)

        with pytest.raises(DatasetNotFoundError):
            loader.load()

    def test_error_message_includes_the_missing_path(self, tmp_path: Path):
        """
        The raised error's message should include the actual path
        that was missing, so a developer immediately knows which
        file to check.
        """
        missing_path = tmp_path / "does_not_exist.nc"
        loader = IMDLoader(missing_path)

        with pytest.raises(DatasetNotFoundError) as exc_info:
            loader.load()

        assert str(missing_path) in str(exc_info.value)
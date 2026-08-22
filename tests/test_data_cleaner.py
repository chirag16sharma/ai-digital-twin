"""
tests/test_data_cleaner.py

Tests for preprocessing/data_cleaner.py — data quality reporting and
cleaning (negative -> 0, NaN -> 0). Uses synthetic_dirty_rainfall_dataset,
which has exactly one known NaN and one known negative value, so
counts and specific cell values can be asserted exactly.
"""

import numpy as np
import pytest
import xarray as xr

from src.exceptions import DatasetSaveError, DatasetSchemaError
from src.preprocessing.data_cleaner import DataCleaner


class TestConstruction:
    """Tests for DataCleaner.__init__."""

    def test_takes_a_copy_not_the_original_object(self, synthetic_rainfall_dataset):
        """
        __init__ should copy the dataset, not store a reference to
        the caller's original — this is the foundation of "cleaning
        never mutates the caller's data."
        """
        cleaner = DataCleaner(synthetic_rainfall_dataset)

        assert cleaner.ds is not synthetic_rainfall_dataset


class TestQualityReport:
    """Tests for quality_report()."""

    def test_prints_correct_total_value_count(
        self, synthetic_dirty_rainfall_dataset, capsys
    ):
        """5 days x 3 lats x 3 lons = 45 total values."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        cleaner.quality_report()

        captured = capsys.readouterr()
        assert "Total Values      : 45" in captured.out

    def test_prints_correct_missing_value_count(
        self, synthetic_dirty_rainfall_dataset, capsys
    ):
        """Fixture has exactly 1 NaN value, at [0, 0, 0]."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        cleaner.quality_report()

        captured = capsys.readouterr()
        assert "Missing Values    : 1" in captured.out

    def test_prints_correct_negative_value_count(
        self, synthetic_dirty_rainfall_dataset, capsys
    ):
        """Fixture has exactly 1 negative value, at [1, 1, 1] = -5.0."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        cleaner.quality_report()

        captured = capsys.readouterr()
        assert "Negative Values   : 1" in captured.out

    def test_clean_dataset_reports_zero_issues(self, synthetic_rainfall_dataset, capsys):
        """
        The CLEAN fixture (not the dirty one) should report 0 missing
        and 0 negative values — confirms quality_report() doesn't
        false-positive on data that's actually fine.
        """
        cleaner = DataCleaner(synthetic_rainfall_dataset)

        cleaner.quality_report()

        captured = capsys.readouterr()
        assert "Missing Values    : 0" in captured.out
        assert "Negative Values   : 0" in captured.out

    def test_raises_dataset_schema_error_when_rainfall_missing(self):
        """A dataset without RAINFALL should raise DatasetSchemaError."""
        dataset = xr.Dataset(coords={"TIME": np.array(["2025-07-01"])})
        cleaner = DataCleaner(dataset)

        with pytest.raises(DatasetSchemaError):
            cleaner.quality_report()


class TestClean:
    """Tests for clean()."""

    def test_nan_value_is_replaced_with_zero(self, synthetic_dirty_rainfall_dataset):
        """The known NaN at [0, 0, 0] should become 0.0 after cleaning."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        result = cleaner.clean()

        assert float(result["RAINFALL"].values[0, 0, 0]) == pytest.approx(0.0)

    def test_negative_value_is_replaced_with_zero(self, synthetic_dirty_rainfall_dataset):
        """The known -5.0 at [1, 1, 1] should become 0.0 after cleaning."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        result = cleaner.clean()

        assert float(result["RAINFALL"].values[1, 1, 1]) == pytest.approx(0.0)

    def test_valid_values_are_left_unchanged(self, synthetic_dirty_rainfall_dataset):
        """
        A cell that was neither NaN nor negative (e.g. [2, 0, 0] =
        20.0, per the deterministic formula) should be untouched by
        cleaning — proves clean() doesn't over-correct valid data.
        """
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        result = cleaner.clean()

        assert float(result["RAINFALL"].values[2, 0, 0]) == pytest.approx(20.0)

    def test_cleaned_dataset_has_no_remaining_nan_values(
        self, synthetic_dirty_rainfall_dataset
    ):
        """After clean(), no NaN values should remain anywhere in the dataset."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        result = cleaner.clean()

        assert not np.isnan(result["RAINFALL"].values).any()

    def test_cleaned_dataset_has_no_remaining_negative_values(
        self, synthetic_dirty_rainfall_dataset
    ):
        """After clean(), no negative values should remain anywhere in the dataset."""
        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)

        result = cleaner.clean()

        assert (result["RAINFALL"].values >= 0).all()

    def test_clean_does_not_mutate_callers_original_dataset(
        self, synthetic_dirty_rainfall_dataset
    ):
        """
        The dataset object originally passed to DataCleaner should
        still contain the NaN/negative values after clean() — since
        __init__ takes a copy, the caller's own reference is
        untouched, even though clean() mutates self.ds in place.
        """
        original_nan_still_present = np.isnan(
            synthetic_dirty_rainfall_dataset["RAINFALL"].values[0, 0, 0]
        )

        cleaner = DataCleaner(synthetic_dirty_rainfall_dataset)
        cleaner.clean()

        assert original_nan_still_present
        assert np.isnan(
            synthetic_dirty_rainfall_dataset["RAINFALL"].values[0, 0, 0]
        )

    def test_raises_dataset_schema_error_when_rainfall_missing(self):
        """A dataset without RAINFALL should raise DatasetSchemaError."""
        dataset = xr.Dataset(coords={"TIME": np.array(["2025-07-01"])})
        cleaner = DataCleaner(dataset)

        with pytest.raises(DatasetSchemaError):
            cleaner.clean()


class TestSave:
    """Tests for save()."""

    def test_saves_dataset_to_valid_path(self, synthetic_rainfall_dataset, tmp_path):
        """save() to a valid, existing directory should succeed and write a file."""
        cleaner = DataCleaner(synthetic_rainfall_dataset)
        output_path = tmp_path / "cleaned.nc"

        cleaner.save(output_path)

        assert output_path.exists()

    def test_raises_dataset_save_error_for_nonexistent_directory(
        self, synthetic_rainfall_dataset, tmp_path
    ):
        """
        Saving to a path whose parent directory doesn't exist should
        raise DatasetSaveError, not a raw FileNotFoundError.
        """
        cleaner = DataCleaner(synthetic_rainfall_dataset)
        bad_path = tmp_path / "nonexistent_subdir" / "cleaned.nc"

        with pytest.raises(DatasetSaveError):
            cleaner.save(bad_path)
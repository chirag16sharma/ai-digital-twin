"""
tests/test_feature_engineer.py

Tests for preprocessing/feature_engineer.py — derived rainfall
features (cumulative sum, rolling averages, lag feature). Tests read
window sizes from config.settings directly, so they stay correct even
if ROLLING_AVERAGE_SHORT_WINDOW / LONG_WINDOW / LAG_FEATURE_DAYS are
changed later.
"""

import numpy as np
import pytest
import xarray as xr

from config.settings import (
    LAG_FEATURE_DAYS,
    ROLLING_AVERAGE_LONG_WINDOW,
    ROLLING_AVERAGE_SHORT_WINDOW,
)
from src.exceptions import DatasetSaveError, DatasetSchemaError
from src.preprocessing.feature_engineer import FeatureEngineer


class TestConstruction:
    """Tests for FeatureEngineer.__init__."""

    def test_takes_a_copy_not_the_original_object(self, synthetic_rainfall_dataset):
        """__init__ should copy the dataset, not store a reference."""
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        assert engineer.ds is not synthetic_rainfall_dataset


class TestCumulativeRainfall:
    """
    Tests for add_cumulative_rainfall(), using the deterministic
    formula at lat_idx=0, lon_idx=0: values are [0, 10, 20, 30, 40]
    across the 5 days, so cumulative sum should be
    [0, 10, 30, 60, 100].
    """

    def test_adds_cumulative_rainfall_variable(self, synthetic_rainfall_dataset):
        """CUMULATIVE_RAINFALL should be added as a new data variable."""
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_cumulative_rainfall()

        assert "CUMULATIVE_RAINFALL" in engineer.get_dataset().data_vars

    def test_cumulative_values_are_correct(self, synthetic_rainfall_dataset):
        """
        At lat_idx=0, lon_idx=0, expected cumulative sum is
        [0, 10, 30, 60, 100].
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_cumulative_rainfall()
        result = engineer.get_dataset()

        point_series = result["CUMULATIVE_RAINFALL"].isel(LATITUDE=0, LONGITUDE=0)
        np.testing.assert_array_almost_equal(
            point_series.values, [0.0, 10.0, 30.0, 60.0, 100.0]
        )

    def test_raises_dataset_schema_error_when_rainfall_missing(self):
        """A dataset without RAINFALL should raise DatasetSchemaError."""
        dataset = xr.Dataset(
            coords={"TIME": np.array(["2025-07-01"], dtype="datetime64[ns]")}
        )
        engineer = FeatureEngineer(dataset)

        with pytest.raises(DatasetSchemaError):
            engineer.add_cumulative_rainfall()


class TestShortWindowAverage:
    """
    Tests for add_short_window_average() — the Day 4 rename of
    add_7day_average(). Uses ROLLING_AVERAGE_SHORT_WINDOW from config
    rather than hardcoding 7, so this test stays correct if the
    config value changes.
    """

    def test_adds_variable_with_dynamic_name(self, synthetic_rainfall_dataset):
        """
        The output variable name should reflect the actual configured
        window size, e.g. "RAINFALL_7DAY_AVG" if
        ROLLING_AVERAGE_SHORT_WINDOW is 7.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_short_window_average()
        result = engineer.get_dataset()

        expected_name = f"RAINFALL_{ROLLING_AVERAGE_SHORT_WINDOW}DAY_AVG"
        assert expected_name in result.data_vars

    def test_first_day_average_equals_first_day_value(self, synthetic_rainfall_dataset):
        """
        Due to min_periods=1, the rolling average on day_idx=0 should
        equal that day's own value (0.0 at lat_idx=0, lon_idx=0) —
        not NaN, since there's no full window yet.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_short_window_average()
        result = engineer.get_dataset()

        variable_name = f"RAINFALL_{ROLLING_AVERAGE_SHORT_WINDOW}DAY_AVG"
        first_day_value = float(
            result[variable_name].isel(LATITUDE=0, LONGITUDE=0, TIME=0).values
        )

        assert first_day_value == pytest.approx(0.0)

    def test_no_nan_values_due_to_min_periods_one(self, synthetic_rainfall_dataset):
        """
        min_periods=1 means every day should have a computed average,
        even near the start of the series where a full window isn't
        available yet — no NaN values should appear anywhere.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_short_window_average()
        result = engineer.get_dataset()

        variable_name = f"RAINFALL_{ROLLING_AVERAGE_SHORT_WINDOW}DAY_AVG"
        assert not np.isnan(result[variable_name].values).any()


class TestLongWindowAverage:
    """
    Tests for add_long_window_average() — the Day 4 rename of
    add_30day_average(). Since the fixture only has 5 days and
    ROLLING_AVERAGE_LONG_WINDOW is typically 30, every day's average
    in this fixture is computed from a partial window — a useful
    edge case in itself.
    """

    def test_adds_variable_with_dynamic_name(self, synthetic_rainfall_dataset):
        """Output variable name should reflect the configured long window size."""
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_long_window_average()
        result = engineer.get_dataset()

        expected_name = f"RAINFALL_{ROLLING_AVERAGE_LONG_WINDOW}DAY_AVG"
        assert expected_name in result.data_vars

    def test_last_day_average_matches_manual_calculation(self, synthetic_rainfall_dataset):
        """
        With only 5 days of fixture data and a long window (e.g. 30),
        the rolling average on the LAST day should simply be the mean
        of all 5 available days at that point — since the window
        never fills, min_periods=1 means it averages whatever exists.
        At lat_idx=0, lon_idx=0: mean([0, 10, 20, 30, 40]) = 20.0.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_long_window_average()
        result = engineer.get_dataset()

        variable_name = f"RAINFALL_{ROLLING_AVERAGE_LONG_WINDOW}DAY_AVG"
        last_day_value = float(
            result[variable_name].isel(LATITUDE=0, LONGITUDE=0, TIME=-1).values
        )

        assert last_day_value == pytest.approx(20.0)


class TestLagFeature:
    """
    Tests for add_lag_feature(), using LAG_FEATURE_DAYS from config.
    At lat_idx=0, lon_idx=0, values are [0, 10, 20, 30, 40]; with a
    lag of 1, PREVIOUS_DAY_RAINFALL at day_idx=2 should equal day_idx=1's
    value (10.0).
    """

    def test_adds_previous_day_rainfall_variable(self, synthetic_rainfall_dataset):
        """PREVIOUS_DAY_RAINFALL should be added as a new data variable."""
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_lag_feature()

        assert "PREVIOUS_DAY_RAINFALL" in engineer.get_dataset().data_vars

    def test_lagged_value_matches_previous_days_actual_value(
        self, synthetic_rainfall_dataset
    ):
        """
        At lat_idx=0, lon_idx=0, day_idx=2's PREVIOUS_DAY_RAINFALL
        should equal day_idx=1's actual rainfall (10.0), assuming
        LAG_FEATURE_DAYS is 1.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_lag_feature()
        result = engineer.get_dataset()

        lagged_value = float(
            result["PREVIOUS_DAY_RAINFALL"]
            .isel(LATITUDE=0, LONGITUDE=0, TIME=1 + LAG_FEATURE_DAYS)
            .values
        )
        actual_previous_value = float(
            result["RAINFALL"].isel(LATITUDE=0, LONGITUDE=0, TIME=1).values
        )

        assert lagged_value == pytest.approx(actual_previous_value)

    def test_first_days_are_nan_due_to_no_prior_data(self, synthetic_rainfall_dataset):
        """
        The first LAG_FEATURE_DAYS time step(s) should be NaN, since
        there's no prior data to reference — documented behavior from
        Day 2's docstring, verified here directly.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_lag_feature()
        result = engineer.get_dataset()

        first_value = result["PREVIOUS_DAY_RAINFALL"].isel(
            LATITUDE=0, LONGITUDE=0, TIME=0
        )

        assert np.isnan(float(first_value.values))


class TestFullFeaturePipeline:
    """
    Tests chaining all four add_*() methods together, matching how
    DataPipeline.run() actually uses this class.
    """

    def test_all_four_features_present_after_full_chain(self, synthetic_rainfall_dataset):
        """
        Calling all four add_*() methods in sequence (as
        DataPipeline does) should result in all four engineered
        variables being present simultaneously.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_cumulative_rainfall()
        engineer.add_short_window_average()
        engineer.add_long_window_average()
        engineer.add_lag_feature()

        result = engineer.get_dataset()

        assert "CUMULATIVE_RAINFALL" in result.data_vars
        assert f"RAINFALL_{ROLLING_AVERAGE_SHORT_WINDOW}DAY_AVG" in result.data_vars
        assert f"RAINFALL_{ROLLING_AVERAGE_LONG_WINDOW}DAY_AVG" in result.data_vars
        assert "PREVIOUS_DAY_RAINFALL" in result.data_vars

    def test_original_rainfall_variable_still_present(self, synthetic_rainfall_dataset):
        """
        Adding engineered features should not remove or overwrite the
        original RAINFALL variable — all four features are additive.
        """
        engineer = FeatureEngineer(synthetic_rainfall_dataset)

        engineer.add_cumulative_rainfall()
        engineer.add_short_window_average()

        result = engineer.get_dataset()

        assert "RAINFALL" in result.data_vars


class TestSave:
    """Tests for save()."""

    def test_saves_dataset_to_valid_path(self, synthetic_rainfall_dataset, tmp_path):
        """save() to a valid directory should succeed and write a file."""
        engineer = FeatureEngineer(synthetic_rainfall_dataset)
        output_path = tmp_path / "features.nc"

        engineer.save(output_path)

        assert output_path.exists()

    def test_raises_dataset_save_error_for_nonexistent_directory(
        self, synthetic_rainfall_dataset, tmp_path
    ):
        """Saving to a nonexistent directory should raise DatasetSaveError."""
        engineer = FeatureEngineer(synthetic_rainfall_dataset)
        bad_path = tmp_path / "nonexistent_subdir" / "features.nc"

        with pytest.raises(DatasetSaveError):
            engineer.save(bad_path)
"""
tests/test_data_pipeline.py

Integration tests for pipeline/data_pipeline.py — the orchestration
class running IMDLoader -> DataExplorer -> DataCleaner ->
FeatureEngineer -> save, end to end. This was the one class in src/
with zero test coverage; these tests exercise the full sequence
against real (temporary, synthetic) files on disk.
"""

from pathlib import Path

import numpy as np
import pytest

from config.settings import ROLLING_AVERAGE_LONG_WINDOW, ROLLING_AVERAGE_SHORT_WINDOW
from src.exceptions import DatasetNotFoundError, PipelineError
from src.pipeline.data_pipeline import DataPipeline


@pytest.fixture
def raw_dataset_path(tmp_path: Path, synthetic_dirty_rainfall_dataset) -> Path:
    """
    Write the DIRTY synthetic dataset (known NaN + negative value) to
    a temp .nc file, as the pipeline's raw input. Using the dirty
    fixture (not the clean one) lets these tests also confirm
    DataCleaner actually ran as part of the sequence, not just that
    the pipeline completes without error.
    """
    path = tmp_path / "raw_input.nc"
    synthetic_dirty_rainfall_dataset.to_netcdf(path)
    return path


class TestDataPipelineRun:
    """Tests for the full run() sequence."""

    def test_completes_successfully_and_returns_dataset(
        self, raw_dataset_path: Path, tmp_path: Path
    ):
        """A valid run should complete and return the final dataset."""
        output_path = tmp_path / "output.nc"
        pipeline = DataPipeline(raw_dataset_path, output_path)

        result = pipeline.run()

        assert result is not None

    def test_output_file_is_written_to_disk(
        self, raw_dataset_path: Path, tmp_path: Path
    ):
        """After run(), the output file should exist on disk."""
        output_path = tmp_path / "output.nc"
        pipeline = DataPipeline(raw_dataset_path, output_path)

        pipeline.run()

        assert output_path.exists()

    def test_result_contains_all_engineered_features(
        self, raw_dataset_path: Path, tmp_path: Path
    ):
        """
        All four FeatureEngineer steps should have run — result
        should have CUMULATIVE_RAINFALL, both rolling averages, and
        PREVIOUS_DAY_RAINFALL, alongside the original RAINFALL.
        """
        output_path = tmp_path / "output.nc"
        pipeline = DataPipeline(raw_dataset_path, output_path)

        result = pipeline.run()

        assert "RAINFALL" in result.data_vars
        assert "CUMULATIVE_RAINFALL" in result.data_vars
        assert f"RAINFALL_{ROLLING_AVERAGE_SHORT_WINDOW}DAY_AVG" in result.data_vars
        assert f"RAINFALL_{ROLLING_AVERAGE_LONG_WINDOW}DAY_AVG" in result.data_vars
        assert "PREVIOUS_DAY_RAINFALL" in result.data_vars

    def test_dirty_input_is_cleaned_by_completion(
        self, raw_dataset_path: Path, tmp_path: Path
    ):
        """
        The input fixture has a known NaN and a known negative value.
        Since DataCleaner runs as Stage 3, the final RAINFALL
        variable should contain neither — proving DataCleaner is
        actually wired into the sequence, not silently skipped.
        """
        output_path = tmp_path / "output.nc"
        pipeline = DataPipeline(raw_dataset_path, output_path)

        result = pipeline.run()

        rainfall_values = result["RAINFALL"].values
        assert not np.isnan(rainfall_values).any()
        assert (rainfall_values >= 0).all()

    def test_raises_pipeline_error_for_missing_input_file(self, tmp_path: Path):
        """
        A nonexistent input file should surface as PipelineError
        (wrapping IMDLoader's DatasetNotFoundError), not propagate
        unwrapped.
        """
        missing_input = tmp_path / "does_not_exist.nc"
        output_path = tmp_path / "output.nc"
        pipeline = DataPipeline(missing_input, output_path)

        with pytest.raises(PipelineError):
            pipeline.run()

    def test_pipeline_error_preserves_original_exception_as_cause(
        self, tmp_path: Path
    ):
        """
        PipelineError should wrap the original DatasetNotFoundError
        as __cause__, not discard it — so the root cause is still
        inspectable, not just "the pipeline failed."
        """
        missing_input = tmp_path / "does_not_exist.nc"
        output_path = tmp_path / "output.nc"
        pipeline = DataPipeline(missing_input, output_path)

        with pytest.raises(PipelineError) as exc_info:
            pipeline.run()

        assert isinstance(exc_info.value.__cause__, DatasetNotFoundError)

    def test_raises_pipeline_error_for_unwritable_output_path(
        self, raw_dataset_path: Path, tmp_path: Path
    ):
        """
        An output path whose parent directory doesn't exist should
        surface as PipelineError (wrapping DatasetSaveError) —
        confirms the earlier fix (routing the save through
        engineer.save() instead of a bare .to_netcdf() call) keeps
        save failures inside the pipeline's unified error handling.
        """
        bad_output_path = tmp_path / "nonexistent_subdir" / "output.nc"
        pipeline = DataPipeline(raw_dataset_path, bad_output_path)

        with pytest.raises(PipelineError):
            pipeline.run()
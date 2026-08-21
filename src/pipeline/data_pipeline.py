"""
pipeline/data_pipeline.py

Orchestrates the full data pipeline for the AI Digital Twin: loading
raw IMD rainfall data, exploring it, cleaning it, engineering
features, and saving the final ML-ready dataset to disk.

This is the only module that wires the ingestion, preprocessing, and
feature engineering stages together — each individual stage
(IMDLoader, DataExplorer, DataCleaner, FeatureEngineer) remains
independently usable and testable in isolation.
"""

from pathlib import Path

import xarray as xr

from src.ingestion.imd_loader import IMDLoader
from src.preprocessing.data_explorer import DataExplorer
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer

from src.exceptions import DigitalTwinError, PipelineError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Runs the end-to-end data pipeline: load → explore → clean →
    engineer features → save.

    Single Responsibility:
        Orchestration. This class does not implement any loading,
        cleaning, or feature logic itself — it delegates each step to
        the appropriate specialized class and controls the order in
        which they run.

    Pipeline flow:
        IMDLoader → DataExplorer → DataCleaner → FeatureEngineer →
        (saved to output_file as rainfall_ai_ready.nc)

    Attributes:
        input_file (Path): Path to the raw IMD NetCDF dataset.
        output_file (Path): Path where the final, feature-engineered
            dataset will be saved.
    """

    def __init__(self, input_file: str | Path, output_file: str | Path) -> None:
        """
        Initialize the pipeline with input and output file paths.

        Args:
            input_file: Path to the raw IMD rainfall NetCDF file.
            output_file: Path where the processed, feature-engineered
                dataset should be saved.
        """
        self.input_file: Path = Path(input_file)
        self.output_file: Path = Path(output_file)

    def run(self) -> xr.Dataset:
        """
        Execute the full pipeline end to end.

        Steps:
            1. Load the raw dataset (IMDLoader).
            2. Print an exploratory summary (DataExplorer).
            3. Report and fix data quality issues (DataCleaner).
            4. Add engineered features (FeatureEngineer):
               cumulative rainfall, short-window average,
               long-window average, previous-day lag.
            5. Save the resulting dataset to self.output_file.

        Returns:
            xr.Dataset: The final, feature-engineered dataset.

        Raises:
            PipelineError: If any stage of the pipeline fails. The
                original exception is preserved as the cause via
                __cause__.
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PIPELINE")
        logger.info(f"Input:  {self.input_file}")
        logger.info(f"Output: {self.output_file}")
        logger.info("=" * 60)

        print("=" * 60)
        print("STARTING DATA PIPELINE")
        print("=" * 60)

        try:
            # Load
            logger.info("Stage 1/5: Loading dataset")
            loader = IMDLoader(self.input_file)
            ds: xr.Dataset = loader.load()

            # Explore
            logger.info("Stage 2/5: Exploring dataset")
            explorer = DataExplorer(ds)
            explorer.summary()

            # Clean
            logger.info("Stage 3/5: Cleaning dataset")
            cleaner = DataCleaner(ds)
            cleaner.quality_report()
            cleaned_ds: xr.Dataset = cleaner.clean()

            # Feature Engineering
            logger.info("Stage 4/5: Engineering features")
            engineer = FeatureEngineer(cleaned_ds)

            engineer.add_cumulative_rainfall()
            engineer.add_short_window_average()
            engineer.add_long_window_average()
            engineer.add_lag_feature()

            feature_dataset: xr.Dataset = engineer.get_dataset()

            # Save
            logger.info("Stage 5/5: Saving final dataset")
            engineer.save(self.output_file)

        except DigitalTwinError as exc:
            logger.error(f"Pipeline failed: {exc}")
            raise PipelineError(
                f"Data pipeline failed: {exc}"
            ) from exc

        print()
        print("Pipeline completed successfully!")
        print(f"Saved to: {self.output_file}")

        logger.info("Pipeline completed successfully!")
        logger.info(f"Saved to: {self.output_file}")

        return feature_dataset
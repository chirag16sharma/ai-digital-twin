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
               cumulative rainfall, 7-day average, 30-day average,
               previous-day lag.
            5. Save the resulting dataset to self.output_file.

        Returns:
            xr.Dataset: The final, feature-engineered dataset — the
                same object that gets saved to self.output_file, so
                callers can continue working with it in memory
                without re-reading it from disk.

        Raises:
            FileNotFoundError: If self.input_file does not exist
                (raised by IMDLoader.load()).
            KeyError: If required variables ("RAINFALL") or
                dimensions ("TIME") are missing from the dataset at
                any pipeline stage.
        """
        print("=" * 60)
        print("STARTING DATA PIPELINE")
        print("=" * 60)

        # Load
        loader = IMDLoader(self.input_file)
        ds: xr.Dataset = loader.load()

        # Explore
        explorer = DataExplorer(ds)
        explorer.summary()

        # Clean
        cleaner = DataCleaner(ds)
        cleaner.quality_report()

        cleaned_ds: xr.Dataset = cleaner.clean()

        # Feature Engineering
        engineer = FeatureEngineer(cleaned_ds)

        engineer.add_cumulative_rainfall()
        engineer.add_7day_average()
        engineer.add_30day_average()
        engineer.add_lag_feature()

        feature_dataset: xr.Dataset = engineer.get_dataset()

        # Save
        feature_dataset.to_netcdf(self.output_file)

        print()
        print("Pipeline completed successfully!")
        print(f"Saved to: {self.output_file}")

        return feature_dataset
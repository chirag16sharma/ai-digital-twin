"""
preprocessing/data_cleaner.py

Responsible for cleaning IMD rainfall data: reporting data quality
issues (missing/negative values), correcting them, and persisting the
cleaned dataset back to disk. Operates on an internal copy of the
dataset so the original, as loaded by IMDLoader, is never mutated.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from src.exceptions import DatasetSaveError, DatasetSchemaError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans an IMD rainfall dataset by handling missing and invalid
    (negative) rainfall values.

    Single Responsibility:
        Data quality — detecting bad values (via quality_report) and
        fixing them (via clean). This class does not add new features
        (FeatureEngineer's job) or explore/summarize the dataset
        (DataExplorer's job).

    Attributes:
        ds (xr.Dataset): A working copy of the dataset being cleaned.
            A copy is taken at construction time so cleaning never
            mutates the dataset the caller originally passed in.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the cleaner with a dataset to clean.

        Args:
            dataset: An xarray Dataset containing a "RAINFALL" data
                variable. A copy is made immediately so the caller's
                original dataset is left untouched.
        """
        self.ds: xr.Dataset = dataset.copy()

    def _get_rainfall(self) -> xr.DataArray:
        """
        Retrieve the "RAINFALL" variable, raising a domain-specific
        exception if it's missing.

        This exists to avoid repeating the same try/except KeyError
        block in quality_report(), clean(), and any future method
        that needs to access RAINFALL.

        Returns:
            xr.DataArray: The RAINFALL data variable.

        Raises:
            DatasetSchemaError: If "RAINFALL" is not present.
        """
        try:
            return self.ds["RAINFALL"]
        except KeyError as exc:
            logger.error("Dataset is missing the 'RAINFALL' variable")
            raise DatasetSchemaError(
                "Dataset is missing the required 'RAINFALL' variable."
            ) from exc

    def quality_report(self) -> None:
        """
        Print a report on data quality issues found in the dataset.

        Reports:
            - Total number of rainfall values
            - Count of missing (NaN) values
            - Count of negative (physically invalid) rainfall values

        Returns:
            None. Console output only — see the note in DataExplorer
            about eventually returning these as structured data
            (e.g. a dict) for programmatic use / testing.

        Raises:
            DatasetSchemaError: If "RAINFALL" is not present in the
                dataset.
        """
        logger.info("Generating data quality report")

        rainfall = self._get_rainfall()

        total_values: int = rainfall.size
        missing_values: int = int(np.isnan(rainfall.values).sum())
        negative_values: int = int((rainfall.values < 0).sum())

        print("=" * 50)
        print("DATA QUALITY REPORT")
        print("=" * 50)

        print(f"Total Values      : {total_values}")
        print(f"Missing Values    : {missing_values}")
        print(f"Negative Values   : {negative_values}")

        if missing_values > 0:
            logger.warning(
                f"{missing_values} missing (NaN) rainfall values found "
                f"({missing_values / total_values:.2%} of dataset)"
            )

        if negative_values > 0:
            logger.warning(
                f"{negative_values} negative rainfall values found "
                f"({negative_values / total_values:.2%} of dataset)"
            )

        logger.info(
            f"Quality report complete. "
            f"Total: {total_values}, Missing: {missing_values}, "
            f"Negative: {negative_values}"
        )

    def clean(self) -> xr.Dataset:
        """
        Clean the rainfall data in place (on the internal copy) and
        return the cleaned dataset.

        Cleaning rules:
            - Negative rainfall values are physically invalid
              (rainfall cannot be less than 0mm) and are replaced
              with 0.
            - Missing (NaN) values are filled with 0, treating a
              missing reading as "no rainfall recorded."

        Returns:
            xr.Dataset: The cleaned dataset (self.ds, after cleaning).
                Note this is the same object as self.ds, not a new
                copy — callers who need to keep an uncleaned reference
                should hold onto their own copy before calling clean().

        Raises:
            DatasetSchemaError: If "RAINFALL" is not present in the
                dataset.
        """
        logger.info("Cleaning dataset")

        rainfall = self._get_rainfall()

        negative_count = int((rainfall.values < 0).sum())
        missing_count = int(np.isnan(rainfall.values).sum())

        rainfall = rainfall.where(rainfall >= 0, 0)
        rainfall = rainfall.fillna(0)

        self.ds["RAINFALL"] = rainfall

        logger.info(
            f"Cleaning complete. "
            f"{negative_count} negative values clipped to 0, "
            f"{missing_count} missing values filled with 0."
        )

        return self.ds

    def save(self, output_path: str | Path) -> None:
        """
        Save the (cleaned) dataset to disk as a NetCDF file.

        Args:
            output_path: Destination path for the .nc file, as a
                string or Path object. Parent directories are not
                created automatically — they must already exist.

        Returns:
            None.

        Raises:
            DatasetSaveError: If the file cannot be written — e.g.
                the parent directory of output_path does not exist.
        """
        output_path = Path(output_path)

        logger.info(f"Saving cleaned dataset to: {output_path}")

        try:
            self.ds.to_netcdf(output_path)
        except FileNotFoundError as exc:
            logger.error(f"Failed to save dataset to {output_path}: {exc}")
            raise DatasetSaveError(
                f"Could not save dataset to {output_path} — "
                f"check that the parent directory exists."
            ) from exc

        print(f"Dataset saved to: {output_path}")
        logger.info(f"Dataset saved successfully to: {output_path}")
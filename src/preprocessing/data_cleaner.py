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
            KeyError: If "RAINFALL" is not present in the dataset.
        """
        rainfall: xr.DataArray = self.ds["RAINFALL"]

        total_values: int = rainfall.size
        missing_values: int = int(np.isnan(rainfall.values).sum())
        negative_values: int = int((rainfall.values < 0).sum())

        print("=" * 50)
        print("DATA QUALITY REPORT")
        print("=" * 50)

        print(f"Total Values      : {total_values}")
        print(f"Missing Values    : {missing_values}")
        print(f"Negative Values   : {negative_values}")

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
            KeyError: If "RAINFALL" is not present in the dataset.
        """
        rainfall: xr.DataArray = self.ds["RAINFALL"]

        rainfall = rainfall.where(rainfall >= 0, 0)

        rainfall = rainfall.fillna(0)

        self.ds["RAINFALL"] = rainfall

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
            FileNotFoundError: If the parent directory of
                output_path does not exist.
        """
        self.ds.to_netcdf(output_path)

        print(f"Dataset saved to: {output_path}")
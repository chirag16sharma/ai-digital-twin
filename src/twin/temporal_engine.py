"""
twin/temporal_engine.py

Responsible for temporal reasoning over the rainfall dataset: detecting
the time coordinate, reporting the available date range, and
retrieving rainfall data for a specific date or date range. This is
the temporal counterpart to SpatialEngine — where SpatialEngine
answers "where," TemporalEngine answers "when."
"""

import numpy as np
import xarray as xr

from config.settings import RAINFALL_VARIABLE_NAME, TIME_ALIASES
from src.exceptions import DatasetSchemaError, InvalidDateError
from src.utils.coordinates import find_coordinate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalEngine:
    """
    Handles temporal queries over an IMD rainfall dataset.

    Single Responsibility:
        Time-based reasoning only — time coordinate detection, date
        range reporting, and date-based rainfall retrieval. Location
        is out of scope here; this class always queries rainfall
        across the *entire spatial grid* for a given date, not for a
        specific point (that combination is QueryEngine's job).

    Attributes:
        dataset (xr.Dataset): The AI-ready rainfall dataset.
        time_name (str): The actual time coordinate name found in
            this dataset (e.g. "TIME" or "time").
        times (np.ndarray): All timestamp values in the dataset,
            cached at construction time.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the Temporal Engine and auto-detect the time
        coordinate.

        Args:
            dataset: AI-ready rainfall dataset, expected to contain a
                time coordinate under one of several possible naming
                conventions (see config.settings.TIME_ALIASES).

        Raises:
            CoordinateNotFoundError: If no recognized time coordinate
                name is found in the dataset.
        """
        self.dataset: xr.Dataset = dataset

        logger.info("Initializing TemporalEngine")

        # Automatically detect the time coordinate, using the alias
        # list defined centrally in config/settings.py — the same
        # constant SpatialEngine uses, so both engines always agree
        # on what "time" can be called.
        self.time_name: str = find_coordinate(self.dataset, TIME_ALIASES)

        # Store all timestamps
        self.times: np.ndarray = self.dataset[self.time_name].values

        logger.info(
            f"TemporalEngine ready. time_name={self.time_name!r}, "
            f"date range=[{self.times[0]}, {self.times[-1]}], "
            f"total days={len(self.times)}"
        )

    def available_dates(self) -> None:
        """
        Print the available date range and total day count for this
        dataset.

        Returns:
            None. Console output only.
        """
        print(f"First Date : {self.first_date()}")
        print(f"Last Date  : {self.last_date()}")
        print(f"Total Days : {self.number_of_days()}")

    def first_date(self) -> np.datetime64:
        """
        Return the earliest date present in the dataset.

        Returns:
            np.datetime64: The first timestamp in the TIME coordinate.
        """
        return self.times[0]

    def last_date(self) -> np.datetime64:
        """
        Return the latest date present in the dataset.

        Returns:
            np.datetime64: The last timestamp in the TIME coordinate.
        """
        return self.times[-1]

    def number_of_days(self) -> int:
        """
        Return the total number of time steps (days) in the dataset.

        Returns:
            int: Count of entries in the TIME coordinate.
        """
        return len(self.times)

    def get_date(self, date: str) -> xr.DataArray:
        """
        Return rainfall data across the full spatial grid for one date.

        Args:
            date: Date string in a format xarray's .sel() can parse
                against the TIME coordinate, e.g. "2025-07-15".

        Returns:
            xr.DataArray: Rainfall values for every grid point on the
                given date (i.e. still indexed by latitude/longitude,
                just fixed at this one date).

        Raises:
            DatasetSchemaError: If the rainfall variable is not
                present in the dataset.
            InvalidDateError: If `date` does not exist in the
                dataset's TIME coordinate.
        """
        try:
            rainfall_var = self.dataset[RAINFALL_VARIABLE_NAME]
        except KeyError as exc:
            logger.error(
                f"Dataset is missing the {RAINFALL_VARIABLE_NAME!r} variable"
            )
            raise DatasetSchemaError(
                f"Dataset is missing the required "
                f"{RAINFALL_VARIABLE_NAME!r} variable."
            ) from exc

        try:
            rainfall: xr.DataArray = rainfall_var.sel(
                {
                    self.time_name: date
                }
            )
        except KeyError as exc:
            logger.error(
                f"Date {date!r} not found in dataset. "
                f"Available range: [{self.first_date()}, {self.last_date()}]"
            )
            raise InvalidDateError(
                f"Date {date!r} not found in dataset. "
                f"Available range: [{self.first_date()}, {self.last_date()}]."
            ) from exc

        return rainfall

    def get_date_range(self, start_date: str, end_date: str) -> xr.DataArray:
        """
        Return rainfall data across the full spatial grid for a range
        of dates (inclusive).

        Args:
            start_date: Start of the range, e.g. "2025-07-01".
            end_date: End of the range, e.g. "2025-07-31". Inclusive,
                per xarray's slice() behavior with label-based
                indexing.

        Returns:
            xr.DataArray: Rainfall values for every grid point across
                all time steps in [start_date, end_date].

        Raises:
            DatasetSchemaError: If the rainfall variable is not
                present in the dataset.
        """
        try:
            rainfall: xr.DataArray = self.dataset[RAINFALL_VARIABLE_NAME].sel(
                {
                    self.time_name: slice(start_date, end_date)
                }
            )
        except KeyError as exc:
            logger.error(
                f"Dataset is missing the {RAINFALL_VARIABLE_NAME!r} variable"
            )
            raise DatasetSchemaError(
                f"Dataset is missing the required "
                f"{RAINFALL_VARIABLE_NAME!r} variable."
            ) from exc

        if rainfall.sizes.get(self.time_name, 0) == 0:
            logger.warning(
                f"Date range [{start_date}, {end_date}] returned no "
                f"data — check the range overlaps the dataset's "
                f"available dates [{self.first_date()}, {self.last_date()}]"
            )

        return rainfall
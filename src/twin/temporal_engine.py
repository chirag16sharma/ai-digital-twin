"""
twin/temporal_engine.py

Responsible for temporal reasoning over the rainfall dataset: detecting
the time coordinate, reporting the available date range, and
retrieving rainfall data for a specific date or date range. This is
the temporal counterpart to SpatialEngine — where SpatialEngine
answers "where," TemporalEngine answers "when."
"""

from typing import List

import numpy as np
import xarray as xr


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
                conventions (see _find_coordinate).

        Raises:
            ValueError: If no recognized time coordinate name is
                found in the dataset.
        """
        self.dataset: xr.Dataset = dataset

        # Automatically detect the time coordinate
        self.time_name: str = self._find_coordinate(
            ["TIME", "time"]
        )

        # Store all timestamps
        self.times: np.ndarray = self.dataset[self.time_name].values

    def _find_coordinate(self, possible_names: List[str]) -> str:
        """
        Find which of several possible coordinate names is actually
        present in the dataset.

        NOTE: This method is currently duplicated verbatim in
        SpatialEngine._find_coordinate(). Flagged in the Day 1 code
        review as duplicated logic — planned to be extracted into a
        shared utility (e.g. src/utils/coordinates.py) so both
        engines call one implementation instead of maintaining two
        copies that could silently drift apart.

        Args:
            possible_names: Candidate coordinate names to check, in
                priority order. The first match found is returned.

        Returns:
            str: The matching coordinate name, exactly as it appears
                in self.dataset.coords.

        Raises:
            ValueError: If none of possible_names exist in the
                dataset's coordinates.
        """
        for name in possible_names:
            if name in self.dataset.coords:
                return name

        raise ValueError(
            f"None of the coordinate names {possible_names} found."
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
            KeyError: If "RAINFALL" is missing, or if `date` does not
                exist in the dataset's TIME coordinate.
        """
        rainfall: xr.DataArray = self.dataset["RAINFALL"].sel(
            {
                self.time_name: date
            }
        )

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
            KeyError: If "RAINFALL" is not present in the dataset.
        """
        rainfall: xr.DataArray = self.dataset["RAINFALL"].sel(
            {
                self.time_name: slice(start_date, end_date)
            }
        )

        return rainfall
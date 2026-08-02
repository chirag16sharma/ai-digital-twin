import numpy as np


class TemporalEngine:
    """
    Temporal Engine for the AI Digital Twin.

    Responsibilities:
    - Detect time coordinates.
    - Display available dates.
    - Retrieve rainfall for a specific date.
    - Retrieve rainfall for a date range.
    """

    def __init__(self, dataset):
        """
        Initialize the Temporal Engine.

        Parameters
        ----------
        dataset : xarray.Dataset
            AI-ready rainfall dataset.
        """

        self.dataset = dataset

        # Automatically detect the time coordinate
        self.time_name = self._find_coordinate(
            ["TIME", "time"]
        )

        # Store all timestamps
        self.times = self.dataset[self.time_name].values

    def _find_coordinate(self, possible_names):
        """
        Find the first matching coordinate name.

        Parameters
        ----------
        possible_names : list
            Possible coordinate names.

        Returns
        -------
        str
            Matching coordinate name.
        """

        for name in possible_names:
            if name in self.dataset.coords:
                return name

        raise ValueError(
            f"None of the coordinate names {possible_names} found."
        )

    def available_dates(self):
        """
        Display the available dates in the dataset.
        """

        print(f"First Date : {self.first_date()}")
        print(f"Last Date  : {self.last_date()}")
        print(f"Total Days : {self.number_of_days()}")

    def first_date(self):
        """
        Return the first available date.
        """

        return self.times[0]

    def last_date(self):
        """
        Return the last available date.
        """

        return self.times[-1]

    def number_of_days(self):
        """
        Return the total number of days.
        """

        return len(self.times)

    def get_date(self, date):
        """
        Return rainfall data for one date.

        Parameters
        ----------
        date : str
            Example: "2025-07-15"

        Returns
        -------
        xarray.DataArray
        """

        rainfall = self.dataset["RAINFALL"].sel(
            {
                self.time_name: date
            }
        )

        return rainfall

    def get_date_range(self, start_date, end_date):
        """
        Return rainfall data between two dates.

        Parameters
        ----------
        start_date : str
        end_date : str

        Returns
        -------
        xarray.DataArray
        """

        rainfall = self.dataset["RAINFALL"].sel(
            {
                self.time_name: slice(start_date, end_date)
            }
        )

        return rainfall
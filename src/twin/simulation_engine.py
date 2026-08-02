import xarray as xr


class SimulationEngine:
    """
    Simulation Engine for the AI Digital Twin.

    Responsibilities
    ----------------
    - Simulate rainfall changes.
    - Keep original dataset unchanged.
    """

    def __init__(self, dataset):
        """
        Initialize the Simulation Engine.
        """

        self.original_dataset = dataset
        self.simulated_dataset = dataset.copy(deep=True)

    def reset(self):
        """
        Reset all simulations.
        """

        self.simulated_dataset = self.original_dataset.copy(deep=True)

    def rainfall_increase(self, percentage):
        """
        Increase rainfall by a percentage.

        Parameters
        ----------
        percentage : float
        """

        factor = 1 + (percentage / 100)

        self.simulated_dataset["RAINFALL"] = (
            self.simulated_dataset["RAINFALL"] * factor
        )

        return self.simulated_dataset

    def rainfall_decrease(self, percentage):
        """
        Decrease rainfall by a percentage.

        Parameters
        ----------
        percentage : float
        """

        factor = 1 - (percentage / 100)

        self.simulated_dataset["RAINFALL"] = (
            self.simulated_dataset["RAINFALL"] * factor
        )

        return self.simulated_dataset

    def dry_spell(self, start_date, end_date):
        """
        Simulate zero rainfall for a date range.
        """

        self.simulated_dataset["RAINFALL"].loc[
            dict(TIME=slice(start_date, end_date))
        ] = 0

        return self.simulated_dataset

    def heavy_rainfall(self, start_date, end_date, multiplier):
        """
        Simulate heavy rainfall.

        Parameters
        ----------
        multiplier : float
            Example:
            2.0 = double rainfall
        """

        self.simulated_dataset["RAINFALL"].loc[
            dict(TIME=slice(start_date, end_date))
        ] *= multiplier

        return self.simulated_dataset

    def get_dataset(self):
        """
        Return simulated dataset.
        """

        return self.simulated_dataset
import xarray as xr


class FeatureEngineer:

    def __init__(self, dataset):
        self.ds = dataset.copy()

    def add_cumulative_rainfall(self):

        self.ds["CUMULATIVE_RAINFALL"] = (
            self.ds["RAINFALL"].cumsum(dim="TIME")
        )

    def add_7day_average(self):

        self.ds["RAINFALL_7DAY_AVG"] = (
            self.ds["RAINFALL"]
            .rolling(TIME=7, min_periods=1)
            .mean()
        )

    def add_30day_average(self):

        self.ds["RAINFALL_30DAY_AVG"] = (
            self.ds["RAINFALL"]
            .rolling(TIME=30, min_periods=1)
            .mean()
        )

    def add_lag_feature(self):

        self.ds["PREVIOUS_DAY_RAINFALL"] = (
            self.ds["RAINFALL"].shift(TIME=1)
        )

    def get_dataset(self):
        return self.ds
    
    def save(self, output_path):

      self.ds.to_netcdf(output_path)

      print(f"Feature dataset saved to: {output_path}")
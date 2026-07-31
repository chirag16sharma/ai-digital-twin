import numpy as np
import xarray as xr


class DataCleaner:

    def __init__(self, dataset):
        self.ds = dataset.copy()

    def quality_report(self):

        rainfall = self.ds["RAINFALL"]

        total_values = rainfall.size
        missing_values = int(np.isnan(rainfall.values).sum())
        negative_values = int((rainfall.values < 0).sum())

        print("=" * 50)
        print("DATA QUALITY REPORT")
        print("=" * 50)

        print(f"Total Values      : {total_values}")
        print(f"Missing Values    : {missing_values}")
        print(f"Negative Values   : {negative_values}")

    def clean(self):

        rainfall = self.ds["RAINFALL"]

        rainfall = rainfall.where(rainfall >= 0, 0)

        rainfall = rainfall.fillna(0)

        self.ds["RAINFALL"] = rainfall

        return self.ds
    
    def save(self, output_path):

     self.ds.to_netcdf(output_path)

     print(f"Dataset saved to: {output_path}")
     
     
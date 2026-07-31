import numpy as np


class DataExplorer:

    def __init__(self, dataset):
        self.ds = dataset

    def summary(self):

        rainfall = self.ds["RAINFALL"]

        print("=" * 50)
        print("IMD DATASET SUMMARY")
        print("=" * 50)

        print(f"Time Steps : {rainfall.sizes['TIME']}")
        print(f"Latitudes  : {rainfall.sizes['LATITUDE']}")
        print(f"Longitudes : {rainfall.sizes['LONGITUDE']}")

        print()

        print("Date Range")
        print(self.ds.TIME.values[0], "to", self.ds.TIME.values[-1])

        print()

        print("Rainfall Statistics")

        print(f"Minimum : {float(rainfall.min()):.2f} mm")
        print(f"Maximum : {float(rainfall.max()):.2f} mm")
        print(f"Average : {float(rainfall.mean()):.2f} mm")
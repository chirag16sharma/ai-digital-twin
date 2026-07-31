from pathlib import Path
import xarray as xr


class IMDLoader:
    """
    Loads IMD rainfall NetCDF datasets.
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load(self):
        """
        Load the NetCDF dataset.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        dataset = xr.open_dataset(self.file_path)

        return dataset  
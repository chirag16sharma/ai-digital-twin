from pathlib import Path

from src.ingestion.imd_loader import IMDLoader
from src.preprocessing.data_explorer import DataExplorer
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer


class DataPipeline:

    def __init__(self, input_file, output_file):

        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

    def run(self):

        print("=" * 60)
        print("STARTING DATA PIPELINE")
        print("=" * 60)

        # Load
        loader = IMDLoader(self.input_file)
        ds = loader.load()

        # Explore
        explorer = DataExplorer(ds)
        explorer.summary()

        # Clean
        cleaner = DataCleaner(ds)
        cleaner.quality_report()

        cleaned_ds = cleaner.clean()

        # Feature Engineering
        engineer = FeatureEngineer(cleaned_ds)

        engineer.add_cumulative_rainfall()
        engineer.add_7day_average()
        engineer.add_30day_average()
        engineer.add_lag_feature()

        feature_dataset = engineer.get_dataset()

        # Save
        feature_dataset.to_netcdf(self.output_file)

        print()
        print("Pipeline completed successfully!")
        print(f"Saved to: {self.output_file}")
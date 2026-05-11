import pathlib

import pandas as pd

from .catalog import Catalog


class Collector:
    """Collects CSV files per mass bin into parquet files.

    The collector expects a scanned `Catalog` instance. For each mass bin the collector
    will read required files (fit, data), optionally read additional result files if
    present, merge them into a single DataFrame and write a parquet file per bin. A
    small manifest describing outputs is returned by `collect()`.
    """

    def __init__(self, catalog: Catalog, output_dir: pathlib.Path) -> None:
        self.catalog = catalog
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # manifest produced by collect()
        self.manifest: pd.DataFrame | None = None

    def collect(self) -> pd.DataFrame:
        """Collect all valid bins from the catalog and write parquet per bin.

        Returns:
            pd.DataFrame: manifest with one row per collected bin containing
                keys: bin_id, parquet_path, rows_in_fit, rows_in_data, rows_merged,
                optional_files, provenance
        """

        return pd.DataFrame()

import pandas as pd

from pwassist.io.binning import BinCollection
from pwassist.preprocessing.preprocessor import (
    Preprocessor,
    PreprocessReport,
    ProcessedBin,
)


class Pipeline:
    """Pipeline for processing PWA results in a bin collection"""

    def __init__(self, preprocessor: Preprocessor | None = None):
        """Initialize the pipeline with a preprocessor (or default steps if None)"""
        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor

    # this has some of the basic idea, but not the full picture. Pipeline needs to
    # combine every aspect (fit ID'ing, preprocessing, assembly, basic plotting, etc.)
    # into a single interface. The pipeline should be the main entry point for the user
    # to run the analysis, and it should handle all the steps in a coherent manner.
    def run(self, collection: BinCollection):
        reports: list[PreprocessReport] = []

        for mass_bin, bundle in collection:
            processed = self.preprocessor.run(bundle)
            reports.append(processed.report)
            yield mass_bin, processed

        total_warnings = sum(len(report.warnings) for report in reports)
        total_time_ms = sum(report.total_time_ms for report in reports)

        print(
            f"Preprocessed {len(reports)} bins with a total of {total_warnings} warnings"
            f" in {total_time_ms:.2f} ms."
        )

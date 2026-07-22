from pwassist.io.binning import BinCollection
from pwassist.preprocessing.preprocessor import Preprocessor, PreprocessReport


def run_pipeline(collection: BinCollection, preprocessor: Preprocessor):
    reports: list[PreprocessReport] = []

    for mass_bin, bundle in collection:
        processed = preprocessor.run(bundle)
        reports.append(processed.report)
        yield mass_bin, processed

    total_warnings = sum(len(report.warnings) for report in reports)
    total_time_ms = sum(report.total_time_ms for report in reports)

    print(
        f"Preprocessed {len(reports)} bins with a total of {total_warnings} warnings"
        f" in {total_time_ms:.2f} ms."
    )

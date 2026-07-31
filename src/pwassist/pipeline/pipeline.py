import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure

from pwassist.core.result import Results
from pwassist.io.binning import BinCollection
from pwassist.io.catalog import Catalog
from pwassist.parser import NamingScheme
from pwassist.preprocessing.preprocessor import (
    DEFAULT_STEPS,
    Preprocessor,
    PreprocessReport,
    PreprocessStep,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineReport:
    """Report for the entire pipeline run"""

    n_bins_found: int
    n_bins_processed: int
    stage_timings_ms: dict[str, float]
    bin_reports: tuple[PreprocessReport, ...]

    @property
    def total_warnings(self) -> int:
        """Total number of warnings across all bins"""
        return sum(len(r.warnings) for r in self.bin_reports)

    @property
    def total_time_ms(self) -> float:
        """Total time taken for the entire pipeline run"""
        return sum(self.stage_timings_ms.values())

    def summary(self) -> str:
        lines = [
            f" Processed {self.n_bins_processed}/{self.n_bins_found} bins,"
            f" with {self.total_warnings} warnings."
        ]
        for stage, ms in self.stage_timings_ms.items():
            lines.append(f"\t{stage}: {ms:.1f} ms")
        for r in self.bin_reports:
            for w in r.warnings:
                lines.append(f"\t [{r.bin_id}] {w}")
        return "\n".join(lines)


@dataclass
class PipelineConfig:
    """All components needed for a pipeline run"""

    root_dir: Path
    naming_scheme: str | NamingScheme = NamingScheme.AUTO
    final_state_parity: int | None = None
    skip_preprocess_steps: Sequence[str] | None = None
    make_plots: bool = True
    plot_output_dir: Path | str | None = None
    coherent_sum_groups: Sequence[str] | None = None  # If none, plot all available
    save_path: Path | None = None
    verbose: bool = False


class Pipeline:
    """Pipeline for processing PWA results in a bin collection

    Runs the full workflow of catalog -> preprocess -> assemble -> plot, and returns a
    PipelineReport with all relevant information.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if config.verbose:
            logging.basicConfig(level=logging.INFO)

    def run(self) -> tuple[Results, PipelineReport]:
        stage_timings: dict[str, float] = {}

        # ---- Step 1: Scan catalog and assemble bin collection ----
        t0 = time.perf_counter()
        catalog = Catalog(self.config.root_dir)
        catalog.scan()
        collection = BinCollection.from_catalog(catalog)
        stage_timings["catalog"] = self._elapsed_time(t0)
        logger.info(f"Found {len(collection)} bins under {self.config.root_dir}.")

        # ---- Step 2: Preprocess the bins ----
        t0 = time.perf_counter()
        preprocessor = Preprocessor(steps=self._resolve_steps())
        processed_bins = [preprocessor.run(bundle) for _, bundle in collection]
        stage_timings["preprocess"] = self._elapsed_time(t0)

        # ---- Step 3: Assemble results from processed bins ----
        t0 = time.perf_counter()
        results = Results.from_processed_bins(
            processed_bins,
            naming_scheme=self.config.naming_scheme,
            final_state_parity=self.config.final_state_parity,
        )
        stage_timings["assemble"] = self._elapsed_time(t0)

        if self.config.save_path is not None:
            t0 = time.perf_counter()
            results.save(self.config.save_path)
            stage_timings["save"] = self._elapsed_time(t0)

        if self.config.make_plots:
            t0 = time.perf_counter()
            self._generate_plots(results)
            stage_timings["plot"] = self._elapsed_time(t0)

        report = PipelineReport(
            n_bins_found=len(collection),
            n_bins_processed=len(processed_bins),
            stage_timings_ms=stage_timings,
            bin_reports=tuple(p.report for p in processed_bins),
        )
        return results, report

    def _resolve_steps(self) -> list[PreprocessStep] | None:
        if not self.config.skip_preprocess_steps:
            return None  # fall back to default steps
        skip = set(self.config.skip_preprocess_steps)
        return [s for s in DEFAULT_STEPS if s not in skip]

    def _generate_plots(self, results: Results) -> None:
        if self.config.plot_output_dir is None:
            logger.warning(
                "Plot output directory not specified. Using current working directory."
            )
            self.config.plot_output_dir = "./"
        out_dir = Path(self.config.plot_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        groups = self.config.coherent_sum_groups or list(results.coherent_sums.keys())
        for group_key in groups:
            if group_key not in results.coherent_sums:
                logger.warning(f"Skipping unknown coherent sum group: {group_key}")
                continue
            ax = results.plot.scan.coherent_sum(group_key)
            fig: Figure | SubFigure | None = ax.get_figure()
            if fig is None:
                logger.warning(
                    f"Could not retrieve figure for group {group_key}. Skipping plot."
                )
                continue
            if fig is SubFigure:
                fig = fig.get_parent()  # Get the parent Figure if it's a SubFigure
            fig.savefig(  # type: ignore
                out_dir / f"coherent_sum_{group_key}.png", dpi=200
            )
            plt.close(fig)  # type: ignore

    def _elapsed_time(self, start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000.0, 3)  # ms

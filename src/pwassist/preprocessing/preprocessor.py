import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Protocol

import pandas as pd

import pwassist.preprocessing.steps as steps
from pwassist.io.binning import BinBundle, MassBin


@dataclass(frozen=True)
class PreprocessReport:
    """Record of steps taken to preprocess PWA results in one bin"""

    bin_id: str
    applied_steps: tuple[str, ...]
    warnings: tuple[str, ...]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        """Return the total time taken for all preprocessing steps."""
        return sum(self.timings_ms.values())


@dataclass(slots=True)
class ProcessedBin:
    """Analysis-ready output of preprocessor steps for a single mass bin."""

    mass_bin: MassBin
    bin_id: str
    report: PreprocessReport
    fit: pd.DataFrame
    data: pd.DataFrame
    correlation: pd.DataFrame | None
    covariance: pd.DataFrame | None
    norm_int: pd.DataFrame | None


class PreprocessStep(Protocol):
    name: str

    def __call__(self, bundle: BinBundle) -> None: ...


@dataclass
class FunctionStep:
    """Wraps plain function into named, pipeline step"""

    name: str
    func: Callable[[BinBundle], None]

    def __call__(self, bundle: BinBundle) -> None:
        self.func(bundle)


DEFAULT_STEPS: list[PreprocessStep] = [
    FunctionStep("check_null_columns", steps.check_null_columns),
    FunctionStep("check_fit_status", steps.check_fit_status),
    FunctionStep("check_error_columns", steps.check_error_columns),
    FunctionStep("wrap_phase_columns", steps.wrap_phase_columns),
    FunctionStep("downcast_numeric_dtypes", steps.downcast_numeric_dtypes),
    FunctionStep("check_covariance_matrix", steps.check_covariance_matrix),
    FunctionStep("check_correlation_symmetry", steps.check_correlation_matrix),
]


class Preprocessor:
    def __init__(self, steps: list[PreprocessStep] | None = None):
        """Initialize the preprocessor with a list of preprocessing steps."""
        self._steps = steps if steps is not None else DEFAULT_STEPS

    @property
    def step_names(self) -> tuple[str, ...]:
        """Return the names of the preprocessing steps."""
        return tuple(step.name for step in self._steps)

    def run(self, bundle: BinBundle) -> ProcessedBin:
        timings: dict[str, float] = {}

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            for step in self._steps:
                start_time = time.perf_counter()
                step(bundle)
                end_time = time.perf_counter()
                timings[step.name] = (
                    end_time - start_time
                ) * 1000  # Convert to milliseconds
            step_warnings = tuple(str(w.message) for w in caught_warnings)

        report = PreprocessReport(
            bin_id=bundle.bin_id,
            applied_steps=self.step_names,
            warnings=step_warnings,
            timings_ms=timings,
        )

        processed = ProcessedBin(
            mass_bin=bundle.mass_bin,
            bin_id=bundle.bin_id,
            report=report,
            fit=bundle.fit.frame,
            data=bundle.data.frame,
            correlation=bundle.correlation.frame if bundle.correlation else None,
            covariance=bundle.covariance.frame if bundle.covariance else None,
            norm_int=bundle.norm_int.frame if bundle.norm_int else None,
        )

        bundle.unload()  # free raw dataframes from memory after processing
        return processed

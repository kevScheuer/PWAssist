import pathlib

import pytest

import pwassist.preprocessing.preprocessor as prep
from pwassist.io.binning import BinBundle, MassBin


class TestPreprocessReport:
    def test_total_time_ms(self):
        report = prep.PreprocessReport(
            bin_id="bin1",
            applied_steps=("step1", "step2"),
            warnings=(),
            timings_ms={"step1": 10.0, "step2": 20.0},
        )
        assert pytest.approx(report.total_time_ms) == 30.0


class TestFunctionStep:
    def test_function_step_calls_func(self, tmp_path: pathlib.Path):
        called = False

        def dummy_func(bundle: BinBundle) -> None:
            nonlocal called
            called = True

        fit_file = tmp_path / "fit_file.csv"

        step = prep.FunctionStep(name="dummy", func=dummy_func)
        bundle = BinBundle(
            mass_bin=MassBin(1.0, 2.0), bin_id="bin1", paths={"fit": fit_file}
        )
        step(bundle)
        assert called


def dummy_step1(bundle: BinBundle) -> None:
    """A dummy preprocessing step that does nothing."""
    pass


def dummy_step2(bundle: BinBundle) -> None:
    """Another dummy preprocessing step that does nothing."""
    pass


class TestPreprocessor:
    custom_steps: list[prep.PreprocessStep] = [
        prep.FunctionStep("check_null_columns", prep.steps.check_null_columns),
        prep.FunctionStep("dummy_step1", dummy_step1),
        prep.FunctionStep("dummy_step2", dummy_step2),
    ]

    def test_step_names(self):
        """Custom steps should be correctly registered in the preprocessor."""
        preprocessor = prep.Preprocessor(steps=self.custom_steps)
        step_names = preprocessor.step_names
        assert len(step_names) == 3
        assert "check_null_columns" in step_names
        assert "dummy_step1" in step_names
        assert "dummy_step2" in step_names

    def test_default_steps(self):
        """Default steps should be correctly registered in the preprocessor."""
        preprocessor = prep.Preprocessor()
        step_names = preprocessor.step_names

        default_step_names = (
            "check_null_columns",
            "check_fit_status",
            "check_error_columns",
            "wrap_phase_columns",
            "downcast_numeric_dtypes",
            "check_covariance_matrix",
            "check_correlation_symmetry",
        )

        assert step_names == default_step_names

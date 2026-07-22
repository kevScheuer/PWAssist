# from __future__ import annotations

# from dataclasses import dataclass, field
# from time import perf_counter

# from steps import PreprocessStep


# @dataclass(frozen=True)
# class PreprocessReport:
#     """Structured details from a pipeline execution."""

#     applied_steps: tuple[str, ...]
#     warnings: tuple[str, ...] = ()
#     timings_ms: dict[str, float] = field(default_factory=dict)


# class PreprocessPipeline:
#     """Composable pipeline that transforms step-by-step."""

#     def __init__(self, steps: list[PreprocessStep]) -> None:
#         self._steps = steps

#     @property
#     def step_names(self) -> tuple[str, ...]:
#         return tuple(step.name for step in self._steps)

#     # def run(self) -> PreprocessReport:
#     #     warnings_acc: list[str] = []
#     #     timings: dict[str, float] = {}

#     #     for step in self._steps:
#     #         start = perf_counter()
#     #         # current, step_warnings = step.apply()
#     #         elapsed_ms = (perf_counter() - start) * 1000.0
#     #         timings[step.name] = round(elapsed_ms, 3)
#     #         warnings_acc.extend(step_warnings)

#     #     report = PreprocessReport(
#     #         applied_steps=self.step_names,
#     #         warnings=tuple(warnings_acc),
#     #         timings_ms=timings,
#     #     )
#         return report

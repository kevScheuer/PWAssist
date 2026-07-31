import argparse
import sys
from pathlib import Path

from pwassist.parser import NamingScheme
from pwassist.pipeline.pipeline import Pipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pwassist",
        description="Catalog, preprocess, and assemble PWA fit results into "
        "a Results object, optionally generating plots.",
    )
    parser.add_argument(
        "root_dir", type=Path, help="Parent dir of mass_X-Y bin subdirectories."
    )
    parser.add_argument(
        "--scheme",
        choices=[s.value for s in NamingScheme],
        default="auto",
        help=(
            "Naming scheme for amplitudes. If not specified, it will attempt to "
            "auto-detect the scheme from a list of recognized schemes."
        ),
    )
    parser.add_argument(
        "--final-state-parity",
        type=int,
        choices=[1, -1],
        default=None,
        help=(
            "Total parity of final state particles (1 or -1). If your naming scheme"
            " does not specify parity, then not setting this will affect your plot"
            " legends."
        ),
    )
    parser.add_argument(
        "--skip-steps",
        nargs="*",
        default=None,
        help="Names of preprocessing steps you would like to skip.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Path to pickle the assembled Results object to.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to write plots to.",
    )
    parser.add_argument(
        "--coherent-sums",
        nargs="*",
        default=None,
        help="Specific coherent sum group keys to plot (default: all).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        root_dir=args.root_dir,
        naming_scheme=args.scheme,
        final_state_parity=args.final_state_parity,
        skip_preprocess_steps=args.skip_steps,
        make_plots=not args.no_plots,
        plot_output_dir=args.plot_dir,
        coherent_sum_groups=args.coherent_sums,
        save_path=args.save,
        verbose=args.verbose,
    )

    try:
        results, report = Pipeline(config).run()
        print(report.summary())
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

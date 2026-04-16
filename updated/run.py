"""CLI entry point for OPU pipeline."""
import argparse
import sys


def cmd_fetch(args):
    from opu.data import fetch_all
    fetch_all(force=args.force)


def cmd_opu(args):
    from opu.uncertainty import build_opu
    build_opu(workers=args.workers)


def cmd_svar(args):
    from opu.svar import run_svar
    run_svar(resume=args.resume, workers=args.workers)


def cmd_figures(args):
    from opu.plotting import generate_figures
    generate_figures(which=args.which)


def cmd_validate(args):
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=False)


def main():
    parser = argparse.ArgumentParser(description="OPU Update Pipeline")
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch)

    p_opu = sub.add_parser("opu")
    p_opu.add_argument("--workers", type=int, default=None,
                       help="SV chain workers (default: all CPUs)")
    p_opu.set_defaults(func=cmd_opu)

    p_svar = sub.add_parser("svar")
    p_svar.add_argument("--resume", action="store_true")
    p_svar.add_argument("--workers", type=int, default=8)
    p_svar.set_defaults(func=cmd_svar)

    p_fig = sub.add_parser("figures")
    p_fig.add_argument("--which", default="all", choices=["all", "opu", "svar", "comparison"])
    p_fig.set_defaults(func=cmd_figures)

    p_val = sub.add_parser("validate")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

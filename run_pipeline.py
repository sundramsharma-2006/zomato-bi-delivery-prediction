"""run_pipeline.py - Reproduce everything from the raw CSVs (FR-10).

    python run_pipeline.py            # full run, ~3-6 minutes on a laptop
    python run_pipeline.py --skip-models

Order: clean -> EDA/KPIs -> significance tests -> delivery model -> churn model
       -> Power BI export + PBIP -> dashboard previews -> PDF reports.
The ML steps use scikit-learn when installed, otherwise the bundled NumPy fallback (src/mini_sklearn.py).
"""
import subprocess, sys, time

STEPS = ["src.clean", "src.eda", "src.stat_checks", "src.model_delivery", "src.model_churn",
         "src.export_powerbi", "src.build_pbip", "src.dashboard_preview", "src.make_reports"]


def main():
    steps = [s for s in STEPS if not (("--skip-models" in sys.argv) and s.startswith("src.model_"))]
    for s in steps:
        t = time.time(); print(f"\n=== python -m {s}")
        r = subprocess.run([sys.executable, "-m", s])
        if r.returncode != 0:
            sys.exit(f"step failed: {s}")
        print(f"--- {s} done in {time.time() - t:.0f}s")
    print("\nAll steps completed. Notebooks (notebooks/*.ipynb) can be re-run independently.")


if __name__ == "__main__":
    main()

# Reproduce and verify the reported earthquake forecasts

This archive contains the essential program code, fixed configurations, fitted parameters, compact empirical probability outputs, and reported summaries. It excludes the original catalog, working manuscripts, figures duplicated in the paper, optimization/debugging logs, solver refinement traces, caches, and synthetic development artifacts. Some imported numerical helper modules retain their historical filenames; they are required runtime dependencies, not empirical evidence.

## Quick verification of saved results

Extract into a new directory. From that directory, use Python 3.12:

    python src/verify_submission.py

The standard-library check verifies every archive checksum, both complete forecast calendars, all 36 horizon/threshold/model mean Brier scores, the primary and four sensitivity paired contrasts, and the widths, quadrature distances, simulation overlap and replicate accounting for all 306 numerical comparisons. It does not run earthquake models. To independently check all binary targets against an already acquired catalog outside the archive, add `--catalog /absolute/path/to/ComCat_catalog.csv`; the file digest is checked first. Sensitivity labels must match the primary calendar. For the fixed-seed block-bootstrap verification, create an environment and install the numerical requirements:

    python3.12 -m venv ../earthquake-numerics
    ../earthquake-numerics/bin/python -m pip install -r requirements-numerics.txt
    ../earthquake-numerics/bin/python src/verify_submission.py --bootstrap

Install the environment outside the extracted archive if using strict file-inventory verification, or keep it in a sibling directory and invoke that interpreter. The verifier ignores Python import caches, Git administrative metadata, and subsequently added root LICENSE, LICENSE.md, LICENSE.txt or .gitignore files. Other unlisted files cause an inventory failure. Recorded files always remain checksum-checked. Numerical bootstrap comparisons allow a tolerance of 1e-13 for floating-point summaries.

## Data and target

The study uses the EarthquakeNPP ComCat snapshot, not a current USGS query. Download it with:

    python src/acquire_earthquakenpp_comcat.py

Source: https://github.com/ss15859/EarthquakeNPP, commit 26d18048e1ca8ff2b02c7016b993de48ed0760f5, file Datasets/ComCat/ComCat_catalog.csv. Expected SHA-256: 5380f71c826e0026634f4e044e18fcbc3e755dd041a031fbc72853e29a748c98. The acquisition script checks the pinned Git blob as well. Follow the original repository and data-provider terms. This archive does not redistribute the catalog or claim rights to its original network data.

There are 6,093 primary-grid cases: 677 weekly origins, three horizons and three thresholds. The primary contrast is seven-day M>=4.5. Four sensitivities each have 677 cases. Outcome-window membership uses exact nanosecond timestamps and strict pre-origin histories. The archived catalog lacks reporting/revision times, so the experiment is retrospective rather than as-issued. Fitting uses the documented datetime/float-day conversion; forecast membership uses integer nanoseconds. Model lags and parameters are serialized decimal inputs.

## Regenerate forecasts

Use a separate working copy for regeneration; the compact evidence archive is intended to remain unchanged. Install requirements-numerics.txt, acquire the catalog, and create an empty results directory before copying the supplied fit subdirectories into it. The supplied pre-test and rate fits can be used directly. Run:

    python src/run_forecasts.py --execution-config experiments/forecast_execution_v1.json
    python src/run_forecasts.py --execution-config experiments/forecast_execution_n099_v1.json
    python src/run_forecasts.py --execution-config experiments/forecast_execution_m035_v1.json
    python src/run_forecasts.py --execution-config experiments/forecast_execution_tighter_v1.json
    python src/run_forecasts_parallel.py --execution-config experiments/forecast_execution_m025_parallel_v1.json

Each command refuses an existing forecast directory unless --resume is given. Resumption verifies the execution signature and skips saved cases. Do not run concurrent commands whose combined worker counts exceed available resources. On the study MacBook Pro M3 Pro (12 CPU cores, 18 GB RAM), the lower-cutoff command uses eight bounded worker processes. Avoid running other forecast commands concurrently with it. GPU acceleration is not used for the outward-arithmetic computations. Budget several hours for the denser lower-cutoff sensitivity; per-evaluation limits may overshoot nominal solver budgets. Resource-limited results remain valid wider intervals when an enclosure is available.

To refit, use a fresh working copy without the corresponding supplied fit output directories:

    python src/fit_etas_configured.py --config experiments/fitting_pretest_v1.json
    python src/fit_etas_configured.py --config experiments/fitting_pretest_n099_v1.json
    python src/fit_etas_configured.py --config experiments/fitting_pretest_m025_v1.json
    python src/fit_etas_configured.py --config experiments/fitting_pretest_m035_v1.json
    python src/fit_rate_baselines.py

Optimizers use twelve deterministic starts. Hardware/library differences may alter fitted parameters or adaptive stopping paths; use the supplied decimal parameter strings to reproduce the conditional forecast definition exactly. Reader configurations retain operative values and original configuration hashes, while omitting historical planning status, unexecuted delay settings and stale descriptive comments. Their execution signatures therefore differ from original project signatures; solver source files are preserved. Compact forecast records retain original output hashes as provenance, not as hashes of regenerated files.

## Meaning of the saved outputs

Probability endpoints are exact rational numbers. Primary probability records contain the observed binary target and all four forecasts. Sensitivity records contain branching and matched-mean probabilities for the same primary-target outcomes. The primary result includes one documented same-input recovery from an already-completed tighter run. Three sensitivity cases have separately documented re-execution of five permission-failed methods; successful original calculations were unchanged. These administrative repairs do not constitute model selection.

Reported numerical intervals condition on model parameters, catalog history, and the specified magnitude law. They do not include parameter or catalog uncertainty. Bootstrap envelopes add a dependence-sensitive sampling summary and are not exact finite-sample coverage guarantees. Numerical comparison outputs use actual catalog-conditioned cases; no synthetic test is presented as an earthquake finding.

## Regenerate figures and the primary score table

In a working copy, install requirements-figures.txt into a separate Python 3.12 environment (its NumPy version differs from the numerical environment). Using that interpreter, run:

    python src/render_empirical_results.py --submission-layout
    python src/render_sensitivity_table.py --submission-layout
    python src/plot_study_design.py
    python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'src'); from plot_reliability import render; render(json.load(open('results/diagnostics.json')),Path('figures/reliability_primary'))"

These commands create figures from the saved empirical summaries; PDF timestamps may differ. Run archive verification before generating additional files. Figure bars and reliability-bin labels have the same definitions as the manuscript.

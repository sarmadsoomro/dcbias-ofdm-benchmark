# Reproducing the study

All commands run from the repository root inside a git checkout. The generation and evaluation
code refuses to write results when files under `dcbias/` have uncommitted changes, because every
result file is stamped with the commit it came from.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.11 or later. Floating-point results can differ in the last digits across BLAS libraries and
platforms; the reported run is recorded in `runs/baseline/provenance.json`.

## Checks that take minutes

```bash
python -m pytest -q                          # 100 passed, 1 skipped (the skipped test needs the twin runs)
python -m dcbias.zhang_check                 # the Zhang and Zhang 2014 Table I check behind baseline B2
python -m dcbias.figures                     # figures/*.pdf from runs/baseline/results_*.json
python -m dcbias.macros runs/baseline -o /tmp/all_numbers.tex   # every number as a LaTeX macro (large)
```

Smoke run of the generator on a tiny design (seconds):

```bash
python -m dcbias.generate --name smoke --set design.n_units_id=4 design.n_train_units=2 \
    design.n_units_ood=1 design.states_per_unit=5 --workers 4
```

## Full reproduction

The order matters: each step reads the previous step's output. Times are for an 8-worker run on a laptop; steps 7, 8, 10 and 11 were not timed separately.

| Step | Command | Time | Writes |
|------|---------|------|--------|
| 1. Dataset | `python -m dcbias.generate --name baseline --workers 8` | about 1.2 h | `runs/baseline/{states.csv, curves.npz, provenance.json}` |
| 2. Main evaluation | `python -m dcbias.evaluate runs/baseline main --workers 8` | about 1.5 h | `results_main.json` |
| 3. Sensing, learning curve, online search | `python -m dcbias.evaluate runs/baseline sensing --workers 8`, then `learning`, then `b8` | about 0.3 h | `results_sensing.json`, `results_learning.json`, `results_b8.json` |
| 4. Sweeps | `python -m dcbias.sweeps --prefix sweep --workers 8` | about 17.5 h | `runs/sweep-*/` |
| 5. Sweep summary | `python -m dcbias.sweep_summary runs sweep- runs/baseline/results_sweeps.json` | seconds | `results_sweeps.json` |
| 6. Misspecification | `python -m dcbias.evaluate runs/baseline misspec --workers 8` | under an hour | `results_misspec.json` |
| 7. Device-shift twins | `python -m dcbias.sweeps --twins --workers 8` | hours | `runs/twin-*/` |
| 8. Device-shift arm | `python -m dcbias.evaluate runs/baseline shift --workers 8 --first-run` | hours | `results_shift.json`, `shift-rows/*.npz` |
| 9. SNR slopes | `python -m dcbias.evaluate runs/baseline slopes --workers 8` | minutes | `results_slopes.json` |
| 10. Post hoc analyses | `python -m dcbias.evaluate runs/baseline b7cal --workers 8`, then `b7wide_imin`, then `python -m dcbias.b7_wide_extras runs/baseline` | hours | `results_b7cal.json`, `results_b7wide_imin.json`, `results_b7wide_extras.json` |
| 11. Counted bits | `python -m dcbias.counted runs/baseline --workers 9`, then `python -m dcbias.counted_bands runs/baseline` | hours | `results_counted.json`, `results_counted_bands.json` |
| 12. Reprocessing | `python -m dcbias.floor_sweepref runs sweep- runs/baseline`, then `python -m dcbias.shift_tails runs/baseline` | minutes | `results_floor_sweepref.json`, `results_shifttails.json` |
| 13. Numbers and figures | `python -m dcbias.macros runs/baseline`, `python -m dcbias.figures` | minutes | `paper_numbers.tex`, `figures/*.pdf` |

Notes:

* `dcbias.shift_tails` reads per-row predictions of an independent refit of the MLP in `runs/baseline/refit-shift/*.npz`. Those files are not part of the repository; the stored `results_shifttails.json` was produced from them.
* `dcbias.macros` emits every leaf of every result file, which exceeds the TeX memory pool. The stored `paper_numbers.tex` holds only the macros that the manuscript cites; the numbers in it are identical to the full emission.
* The pre-run gates (`python -m dcbias.gates crn --n 200 --workers 8`, `python -m dcbias.gates interp --run runs/baseline`) check the common-random-numbers label noise, the counted-bit agreement and the oracle interpolation error before a full run.

## Determinism

Every simulated unit draws from its own random streams, so a subset of units regenerates the same rows as the full
design (`--splits`, `--limit`). The seed and every stream index are in `dcbias/config.json` and `dcbias/physics.py`.
Model fits use fixed seeds; the learned controllers are reported as the mean of five seeds.

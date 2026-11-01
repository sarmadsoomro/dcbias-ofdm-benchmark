# dcbias-ofdm-benchmark

Simulator, controllers, evaluation code and stored results for the manuscript

> **Learned or Model-Based DC-Bias Control for DCO-OFDM LiFi? A Simulation Benchmark Under LED Model Error and Device Shift**
> Sarmad Soomro, Shreepragatheesh Govindaraju, Asif Ali Wagan (submitted).

The study asks whether a learned DC-bias predictor beats a non-learning controller that uses the same
observations and a physical model of the LED, when both are scored by the bit-error rate (BER) of the link
and not by $R^2$ against a simulator label. Controllers are scored by **BER regret**: the base-10 logarithm of
the BER at the commanded bias over the BER at the best bias for that state.

## What is here

| Path | Content |
|------|---------|
| `dcbias/` | Python package: electrothermal DCO-OFDM link physics, data generation, baselines B0 to B8, learned controllers, evaluation, sweeps, device-shift arm, counted-bit check, figure and number-macro generators |
| `dcbias/config.json` | Every simulator parameter with units, source DOI and verification basis, plus design notes |
| `tests/` | 101 unit and regression tests (about 45 s) |
| `runs/baseline/` | Stored results of the reported run: `results_*.json`, `states.csv` (all simulated states and labels), `provenance.json`, `paper_numbers.tex` |
| `runs/sweep-*/` | Result summary and provenance of each one-factor-at-a-time sweep run |
| `figures/` | The manuscript figures, regenerated from the stored results |
| `docs/` | `reproducing.md` (run order, cost, determinism) and `results.md` (what each result file holds and where the paper uses it) |

Not in the repository, because of size (several hundred megabytes): the raw oracle and label curves, the per-row prediction
arrays of the device-shift and counted-bit analyses, and the device-shift twin runs. They are regenerated
from the fixed seed by the commands in `docs/reproducing.md`. A deposit with a DOI is planned on acceptance of
the manuscript.

## Quick start

```bash
git clone https://github.com/sarmadsoomro/dcbias-ofdm-benchmark.git
cd dcbias-ofdm-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 100 passed, 1 skipped
python -m dcbias.figures            # rebuilds figures/*.pdf from runs/baseline/results_*.json
```

The code requires Python 3.11 or later. The reported run used numpy 2.2.6, scipy 1.16.0, scikit-learn 1.7.0
and Python 3.11.9 (recorded in `runs/baseline/provenance.json`).

## Reading the results without running anything

* `runs/baseline/results_main.json`: mean regret of every controller on the four test sets, the confirmatory contrast, hyperparameters, latency and size.
* `runs/baseline/results_shift.json`: the device-shift arm (cost of each LED shift per controller).
* `runs/baseline/states.csv`: one row per simulated state (exogenous state, latent device parameters prefixed `u_`, labels).
* `docs/results.md` lists every file.

## Controllers

| Name | Inputs | Rule |
|------|--------|------|
| B0 | none | Constant bias minimising mean training regret |
| B1a, B1b, B1c | clip level estimate | Published `k * sigma_I` rules and a fitted `k` |
| B2 = B3 | clip level estimate | Midpoint of the clipping window (B2 credits the objective of Zhang and Zhang 2014, whose Table I is not reproducible; see `dcbias/zhang_check.py`) |
| B4, B5, B6a | state | Data-binned mean of the training label |
| B6b | state | Nominal-device model tabulated on a regular grid; no training labels |
| B7 | state and probe observables | Bayesian calibrate-then-optimise over LED curvature, droop and efficiency; bias minimising expected log BER |
| B7w | adds waveform tap | B7 with a ceiling estimate from the waveform maximum (exploratory) |
| B8 | live link | Golden-section search on the live link; excluded from selection |
| Ridge, RF, HistGB, MLP | state, or state plus observables | Regression on the training label |

## Conventions

* `runs/<name>/` is one dataset plus its results. `sweep-<key>=<value>` is a sweep run and `twin-<name>` a device-shift twin of the test units.
* Provenance fields named `sha` (in `provenance.json` and `_eval_provenance`) are commit hashes of the development history that produced the stored results. That history is not published; the code in this repository is the final state of the same code.
* Results are written by the code, and the manuscript quotes them through generated macros (`dcbias/macros.py`); no number in the paper is typed by hand.

## Licence and citation

Code: MIT (`LICENSE`). Results and figures: CC BY 4.0 (`DATA_LICENSE.md`). Cite via `CITATION.cff`.

## Contact

Sarmad Soomro, School of Information Technology, Whitecliffe College, Christchurch, New Zealand, `sarmads@whitecliffe.ac.nz`.

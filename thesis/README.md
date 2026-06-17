# Thesis code & artifacts

*Beyond GMV: Welfare Incidence and Context-Dependent Method Selection in Ride-Hailing
Control under Heterogeneous Drivers.*

The thesis itself is **`THESIS.md`**. The design spec is `DESIGN.md`; the bibliography is
`REFERENCES.md`.

## Layout

```
thesis/
  THESIS.md          the thesis
  DESIGN.md          engineering/scientific spec the code implements
  REFERENCES.md      bibliography (calibration + corpus sources)
  src/               all code
  results/           generated CSVs + validation.json + analysis_digest.txt
  figures/           generated PNGs
```

## Source modules (`src/`)

| file | what it is |
|---|---|
| `config.py` | economic constants (independently sourced) + `ScenarioConfig` |
| `simulator.py` | the zone-based two-sided market simulator + welfare accounting |
| `methods.py` | the 27-controller library (3 matching × 3 pricing × 3 rebalancing) |
| `engine.py` | value-table pretraining + single-combo runner |
| `scenarios.py` | scenario generator (LHS + named corner cases) + elasticity→σ map |
| `validate.py` | validity checks V1 (elasticity), V2 (square-root law), V3 (Castillo) |
| `run_experiments.py` | main grid: every (scenario × combo × seed) → `results/results.csv` |
| `selection.py` | oracle, regret, context→method selector, baselines |
| `analysis.py` | results-chapter summary tables → `results/*.csv` |
| `figures.py` | all thesis figures → `figures/*.png` |

## Reproduce

```bash
cd thesis/src
pip install numpy scipy pandas scikit-learn matplotlib seaborn

python validate.py            # V1-V3 + Castillo regime sweep  (~3 min)
python run_experiments.py --lhs 60 --seeds 8 --procs 4   # main grid (~8 min)
python selection.py           # regret table (RQ3) + objective divergence (RQ4)
python analysis.py            # summary tables for the results chapter
python figures.py             # all figures
```

All randomness is seeded; results are deterministic given the seeds. Economic constants
are in `config.py` with citations; Castillo (2025) is used only as a validation target in
`validate.py`, never as a model input.

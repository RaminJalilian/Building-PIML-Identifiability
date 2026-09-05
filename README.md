# Building-PIML-Identifiability

The code, synthetic data, and results supporting the paper *Physics-Informed Neural
Calibration of RC Building Energy Models: Prediction Accuracy versus Parameter
Identifiability* by Ramin Jalilian and Ehsan Kamel.

## Layout

```
Building-PIML-Identifiability/
├── src/        simulation, calibration variants, experiment grid, figure scripts
├── data/       synthetic datasets and the S1/S2/S3 sensing views
├── results/    run records and summary tables
└── figures/    manuscript figures and the schematic source
```

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPython 3.10.12
```

## Regenerating

```bash
python src/simulate.py        # -> data/
python src/experiment.py      # -> results/   (~15 min)
python src/make_figures.py    # -> figures/v4/
```

These commands overwrite the committed data, results, and figures. Scripts anchor their
paths to the repository root and can be run from any working directory.

## Integrity

SHA-256 hashes for every committed dataset and results file are recorded in
[`INTEGRITY.md`](INTEGRITY.md).

## Citation

> Jalilian, R., and Kamel, E. (2026). *Physics-Informed Neural Calibration of RC Building
> Energy Models: Prediction Accuracy versus Parameter Identifiability.* Manuscript in
> preparation.

## License

MIT — see [`LICENSE`](LICENSE). Copyright © 2026 Ramin Jalilian and Ehsan Kamel.

# Integrity Manifest — Authoritative Artifacts

Records the SHA-256 and generating provenance of every committed authoritative data/result
artifact, so a clean checkout can be verified and reproduced. Regenerate with:

```bash
python src/simulate.py       # regenerates data/*.csv (deterministic, seed=0)
python src/experiment.py     # regenerates results/results_all.json + result CSVs (270 runs)
python src/experiment.py --reaggregate   # re-derives identifiable-metric CSV columns only
```

Repository contract: `data/*.csv` and `results/results_all.json` are tracked; other
generated files under `data/`, `results/`, and `figures/` (except `figures/v4/` and the
hand-drawn `figures/Figure1.drawio`) are ignored.

Generating commits: dataset + raw results generated at `5f66392` (post weather-fix `5b10428`);
the identifiable-metric columns of the aggregated/seeds CSVs were later derived via
`--reaggregate` (no run value changed).

| File | bytes | SHA-256 |
|---|---:|---|
| `data/synthetic_dataset.csv` | 475463 | `0beb6e9cfef0233f8ae314c6134730e54bf1347fe3019209b77346a1797d54d7` |
| `data/sensing_S1.csv` | 316934 | `641ba937b556068876ec6f4c0bd0af781b1bf93004fbb1e07c2d76ad0f66f439` |
| `data/sensing_S2.csv` | 320481 | `21cd74781c484cc3b6ef226cedabeb66e8270222bfa6c0a9266cc05eca2a5263` |
| `data/sensing_S3.csv` | 354740 | `82bf5f099ec840bcd8d36c5dc5f4b40263f67d7808d788f7759c54cafa5d051f` |
| `results/results_all.json` | 8303402 | `8951095b526da8837b7534e4fa926467cc498f4b12b515329595f9ae4968e32f` |
| `results/results_aggregated.csv` | 11914 | `6ca4462189c80f2eb5e8f6b18c071ad3e386d6aa2cc5dee2c14d422eb841dbd1` |
| `results/results_summary_seeds.csv` | 61978 | `00d401c963f591f792b099a087591aff2763b55ed493a70308343419ee0a0cb8` |
| `results/results_summary.csv` | 10111 | `b4c1d9d321a6d0212d5749f5f341d1884fdff211556542a1c68d442cc6ec3a9b` |

Ground-truth identifiable combinations: G_s* = 1/(R_zw+R_wo) = 307.692 W/K;
G_inf* = 1/R_inf = 2857.143 W/K. Only R_s=R_zw+R_wo, G_s, R_inf, G_inf are identifiable.

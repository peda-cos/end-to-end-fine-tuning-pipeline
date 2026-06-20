# Component licenses

This repository is a portfolio implementation. No third-party model weights or
benchmark datasets are redistributed.

| Component | Path | License |
|---|---|---|
| Pipeline source code, configs, scripts, dashboard | `src/`, `configs/`, `dashboard/`, `*.sh` | MIT (see `LICENSE`) |
| Generated synthetic training/evaluation data | `fixtures/`, `outputs/data/` | CC0-1.0 |
| Trained model parameters and adapters | `outputs/models/`, `outputs/quant/` | CDLA-Permissive-2.0 |
| Documentation, README, model card, data card | `README.md`, `outputs/model-card.md`, `outputs/data-card.md` | CC-BY-4.0 |

Generated model artifacts are produced locally by the pipeline and are not
committed. The emitted GPU configs reference third-party models and tools whose
licenses are governed by their upstream projects.

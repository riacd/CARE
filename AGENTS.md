# Repository Guidelines

## Project Structure & Module Organization
`processed_data/` and `splits/` contain the packaged datasets used by the benchmark. `task1_baselines/` and `task2_baselines/` hold runnable baselines, example outputs in `results_summary/`, and method-specific notebooks or helper scripts. `CREEP/` is the installable model package for Task 2 pretraining and embedding extraction. `generate_datasets_splits/` contains the notebooks and utilities used to reproduce dataset curation. Figures live in `figs/`, and `performance_evaluation.ipynb` is the main analysis notebook.

## Build, Test, and Development Commands
Create the lightweight analysis environment with `conda create -n CARE_processing python=3.8 -y` and `pip install -r requirements.txt`. Install the model package only when working on CREEP: `cd CREEP && pip install -e .`.

Common entry points:
- `python task1_baselines/rank_tabulate_random.py`: generate Task 1 random baseline CSVs.
- `python task1_baselines/BLAST/run_diamond_blast.py`: run the BLAST baseline.
- `python task2_baselines/rank_tabulate_random.py`: generate Task 2 random baseline CSVs.
- `python task2_baselines/downstream_retrieval.py --pretrained_folder=... --query_dataset=easy_reaction_test --reference_dataset=all_ECs --query_modality=reaction --reference_modality=protein`: run Task 2 retrieval.
- `jupyter notebook performance_evaluation.ipynb`: compute benchmark metrics from output CSVs.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, snake_case for functions, variables, files, and CLI flags, and simple top-level scripts guarded with `if __name__ == "__main__":` where appropriate. Keep new scripts near the task they support (`task1_baselines/`, `task2_baselines/`, or `generate_datasets_splits/`). Prefer small, dependency-light utilities; this repo does not use a formatter or lint config.

## Testing Guidelines
There is no dedicated `tests/` directory or CI suite. Validate changes by running the affected script or notebook on the smallest relevant split and checking regenerated artifacts under `results_summary/` or method-specific `output/` folders. For Python-only edits, add a quick syntax smoke test such as `python -m py_compile path/to/file.py`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Update README.md`; keep commit titles concise and specific. In pull requests, state the task affected, list the commands you ran, note any regenerated data files, and attach screenshots only for notebook or figure changes. Avoid committing large derived artifacts unless the update intentionally refreshes benchmark outputs.

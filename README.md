# MOOSE-FS

[![tests](https://github.com/arthurbabey/ensemblefs/actions/workflows/tests.yml/badge.svg)](https://github.com/arthurbabey/ensemblefs/actions/workflows/tests.yml)
[Documentation](https://arthurbabey.github.io/ensemblefs/)

## Overview

MOOSE-FS is a feature selection library that leverages an ensemble-based approach to optimize both predictive performance and stability. By combining multiple feature selection methods, merging strategies, and evaluation metrics, it provides a highly flexible and tunable pipeline for both classification and regression tasks. The package automates feature selection across multiple iterations and uses Pareto optimization to identify the best feature subsets.

Users can define their feature selection process by:
- Selecting feature selection methods from predefined options or implementing custom ones.
- Choosing merging strategies to aggregate feature rankings.
- Specifying performance metrics to evaluate selected features.
- Configuring the number of features to select and the number of repetitions.
- Working with either **classification** or **regression** problems.

The library allows defining feature selectors, merging strategies, and metrics either as **class instances** or as **string identifiers**, which act as placeholders for built-in methods. The framework is modular and can be easily extended by adding new selection algorithms or merging strategies.

---

## Requirements

- **Python** 3.9 or higher
- **Dependencies**: Automatically installed from `pyproject.toml`.

---

## Installation

### From PyPI (Recommended)

```bash
pip install moose-fs
```

### From Source

To install the package from source, run:

```bash
pip install git+https://github.com/arthurbabey/ensemblefs.git
```

Alternatively, clone the repository and install locally:

```bash
git clone https://github.com/arthurbabey/ensemblefs.git
cd ensemblefs
pip install .
```

---

## Using the Library

### 1. Feature Selection Pipeline

The core of MOOSE-FS is the `FeatureSelectionPipeline`, which provides a fully configurable workflow for feature selection. Users can specify:
- Feature selection methods
- Merging strategy
- Evaluation metrics
- Task type (classification or regression)
- Number of features to select
- Number of repetitions

#### Example Usage

```python
# `data` can be a single DataFrame (last column = target)
# or you can pass `X` and `y` separately.
# Assume `data` is a pandas DataFrame whose last column "label" holds the targets.
from moosefs import FeatureSelectionPipeline

fs_methods = ["f_statistic_selector", "random_forest_selector", "svm_selector"]
merging_strategy = "union_of_intersections_merger"

pipeline = FeatureSelectionPipeline(
    X=data.drop(columns=["label"]),
    y=data["label"],
    fs_methods=fs_methods,
    merging_strategy=merging_strategy,
    num_repeats=5,
    task="classification",
    num_features_to_select=10,
    stability_mode="fold_stability",  # Options: "selector_agreement", "fold_stability", "all"
)

# Run the pipeline
selected_features, best_ensemble = pipeline.run()
```

This will run feature selection using K-fold cross-validation, merge results using the chosen strategy, and return the best-selected features after refitting on the full dataset.

#### Stability Modes

The `stability_mode` parameter controls which stability metrics are included in the Pareto optimization:

- `"selector_agreement"`: Measures agreement between selectors within each ensemble
- `"fold_stability"`: Measures consistency of selected features across CV folds (default)
- `"all"`: Includes both stability metrics in the optimization

### 2. Extensibility

MOOSE-FS is designed to be easily extended. Users can implement custom:
- **Feature selection methods**: Define a new feature selector class and integrate it into the pipeline.
- **Merging strategies**: Implement a custom strategy to aggregate selected features.
- **Metrics**: Add new evaluation metrics tailored to specific tasks.

New methods can be used directly in the pipeline by passing the class or a corresponding identifier.

---

## Using the CLI

Once installed, the pipeline can also be run from the command line using:

```bash
efs-pipeline
```

This command executes `scripts/main.py` using parameters from `scripts/config.yaml`. Users can specify a different config file:

```bash
efs-pipeline path/to/your_config.yaml
```

### Example `config.yaml`

```yaml
experiment:
  name: "example_experiment"
  results_path: "results/"
  data_path: "data/input_data.csv"

preprocessing:
  normalize: true
  handle_missing: true

pipeline:
  fs_methods: ["f_statistic_selector", "random_forest_selector"]
  merging_strategy: "union_of_intersections_merger"
  num_repeats: 5
  task: "classification"
  num_features_to_select: 10
```

### Results

The results are saved in a structured directory under `results/example_experiment/`, including:
- A **text file** summarizing the pipeline run.
- A **CSV file** containing the final results.

---

## Code Structure

- **`core/`**: Core modules for data processing, metrics, and stability computation.
- **`feature_selection_pipeline.py`**: Defines the main feature selection workflow.
- **`feature_selectors/`**: Implements feature selection methods (e.g., F-statistic, mutual information, RandomForest, SVM).
- **`merging_strategies/`**: Implements merging strategies such as Borda count and union of intersections.

---

## Contributing

Contributions are welcome! If you have ideas for improving MOOSE-FS, feel free to open an issue or submit a pull request.

### Development (uv)

This project uses uv for local environments and dependency management. The library builds via the existing PEP 517 backend (hatchling); uv only manages the environment, installs, and command execution.

- Install/select Python 3.9+ and ensure `uv` is installed.
- Create a local virtual environment in `.venv`:

```bash
uv venv --python 3.9
```

- Install dev dependencies (editable):

```bash
uv pip install -e ".[dev]"
```

- Install pre-commit hooks:

```bash
uv run pre-commit install
```

- Run formatting and linting:

```bash
uv run ruff format .
uv run ruff check --fix .
```

- Run tests:

```bash
uv run pytest -q
```
---

## License

This project is licensed under the MIT License.

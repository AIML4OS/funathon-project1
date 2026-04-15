# Funathon Project 1 – Applying Machine Learning to Tabular Data

> An end-to-end machine learning pipeline for **fine-grained housing price prediction** in France, from raw data preprocessing to production deployment.

📖 **Full documentation:** [aiml4os.github.io/funathon-project1](https://aiml4os.github.io/funathon-project1/)

## Overview

This project walks through the complete lifecycle of a machine learning application applied to real estate tabular data. Using synthetic data reproducing the French **DVF+ (Demandes de Valeurs Foncières)** and land registry dataset, the goal is to build a model that predicts housing prices at a fine geographic level, and then deploy it as a production-ready API.

The project is structured as a progressive, hands-on tutorial organized in five parts:

1. **Data description** — understanding the input variables
2. **Pre-processing** — cleaning and preparing the dataset
3. **ML model training** — training and comparing gradient boosting models
4. **Model logging** — loggingt.
4. **Model deployment** — API deployment.


## Project Structure

```
.
├── starting_point/          # Notebooks with exercises (to be completed)
├── intermediate_solutions/  # Step-by-step partial solutions
├── solution/                # Full reference solutions
├── 1-preprocessing.qmd      # Part 1: Data preprocessing
├── 2-GB_model.qmd           # Part 2: Gradient Boosting model
├── 2-RF_model.qmd           # Part 2: Random Forest model
├── 3-metrics.qmd            # Part 3: Model evaluation
├── 4-logging.qmd            # Part 4: MLFlow experiment tracking
├── 5-deployment.qmd         # Part 5: FastAPI deployment
├── pyproject.toml           # Python dependencies (managed with uv)
└── _quarto.yaml             # Quarto site configuration
```

---

## The Five Parts

### 1. Data Description

The dataset is based on synthetic data mimicking the **DVF+** (French land registry transactions) and land registry. Each row represents a single real estate sale and contains ~47 variables describing the property, the transaction, and its geographic location.

Key variables include:

| Variable | Description |
|---|---|
| `valeurfonc` | Transaction value (target) |
| `dsupdc` | Floor area (m²) |
| `dteloc` | Property type (1 = house, 2 = flat) |
| `depcom` | Municipality code |
| `x`, `y` | Geographic coordinates |
| `dnbppr` | Number of main rooms |
| `dnbcha` | Number of bedrooms |
| `jannath` | Year of construction |
| `anneemut` | Year of transaction |
| `distance_ltm` | Distance to the coastline |
| `nb_garages`, `nb_piscines`, `nb_terrasses`, ... | Outbuildings and amenities |

> See the full variable dictionary in [the dedicated page](intro_data.Qmd).

### 2. Pre-processing

**Tools:** [`pandas`](https://pandas.pydata.org/docs/user_guide/index.html)

This step covers:
- Handling missing values and outliers (e.g. filtering extreme `price_per_sqm` values)
- Selecting a relevant feature subset from the 47 available variables
- Computing derived features such as `price_per_sqm`
- Exploratory data analysis with `seaborn.pairplot` and `pandas.DataFrame.hist`


### 3. Training a ML Model

**Tools:** [`scikit-learn`](https://scikit-learn.org/stable/user_guide.html)

Reference: [INSEE working document on bagging and boosting methods](https://inseefrlab.github.io/DT_methodes_ensemblistes/), [`crospint`](https://pypi.org/project/crospint/)

Goals:
- Split data into training and test sets
- Train and compare three models: `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`
- Explore location encoding strategies (One-Hot Encoding, native categorical support)
- Hyperparameter tuning via cross-validation (grid search, random search, optionally Optuna)
- Apply early stopping and metric logging during training
- Evaluate models using **MAPE** and **R²**


### 4. & 5. Model logging and deployment

**Tools:** [`MLFlow`](https://mlflow.org/docs/latest/ml/), [`FastAPI`](https://fastapi.tiangolo.com/tutorial/)

This part covers:
- **Experiment tracking** with MLFlow: saving all runs, models, and associated metrics
- **API deployment** with FastAPI: expose a prediction endpoint that takes property attributes (surface, location, number of rooms, etc.) and returns a predicted price


## Getting Started

### Prerequisites

- Python >= 3.13
- [`uv`](https://docs.astral.sh/uv/)
- [SSPCloud account](https://datalab.sspcloud.fr/)  (recommended)
- GitHub account (recommended)

### Installation of this repo

```bash
# Fork the repository
git clone https://github.com/AIML4OS/funathon-project1.git
cd funathon-project1

# Install dependencies with uv
uv sync
```

### Running the notebooks

```bash
# Render the full Quarto website locally
uv run quarto render

# Or preview it
uv run quarto preview
```

## Contributing

Contributions are welcome! Whether you spotted a bug, a typo, an outdated dependency, or have an idea to improve the tutorial, here's how to get involved:

1. **Open an issue** — head to the [Issues tab](https://github.com/AIML4OS/funathon-project1/issues) and describe what you found or what you'd like to see. Please check that a similar issue doesn't already exist before opening a new one.
2. **Submit a Pull Request** — fork the repository, make your changes on a dedicated branch, and open a PR against `main`. Briefly describe what you changed and why. Linking the PR to the relevant issue is appreciated.

No contribution is too small — fixing a broken link or clarifying a comment is just as valuable as adding a new feature.


## About

This project was developed as part of the **AIML4OS Funathon** — a collaborative hackathon focused on applying AI and machine learning methods to open statistical data.

🔗 [AIML4OS Organization](https://github.com/AIML4OS)

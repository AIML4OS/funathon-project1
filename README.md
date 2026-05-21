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
├── subject/                 # Quarto files for website
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
| `price` | Transaction value (target) |
| `farea` | Floor area (m²) |
| `prop_type` | Property type (1 = house, 2 = flat) |
| `prop_loc_citycode` | Municipality code |
| `prop_loc_x`, `prop_loc_y` | Geographic coordinates |
| `n_mrooms` | Number of main rooms |
| `n_slr` | Number of bedrooms |
| `prop_year_harm` | Year of construction |
| `trans_year` | Year of transaction |
| `dist_tosea` | Distance to the coastline |
| `n_garage`, `n_pool`, `n_terrace`, ... | Outbuildings and amenities |

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
- MLFlow

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

## Data
Data are synthetic data. 

French version of the data is stored in two files in the `projet-funathon/2026/project1/data/0_raw/` folder : `2026/project1/data/0_raw/transactions_flats_FR_raw.parquet` and `2026/project1/data/0_raw/transactions_houses_FR_raw.parquet`.

The script to convert French labelled data to English is stored in `temp/0_generate_input.py`.

## Admin 
A script transforms .qmd files from `subject/file.qmd` into `intermediate_solutions/script.py`. 
The script extracts all code in the listed qmd files that are in **executable code cells**, meaning starting with the exact set of character: \`\`\`\{python.
Code cell starting with \`\`\` python for example won't be pasted into the intermediate solution scripts.

To run it : 

``` bash
bash solution/admin/extract_all.sh false  # extract and not testing scripts
bash solution/admin/extract_all.sh        # extract and not testing scripts
bash solution/admin/extract_all.sh true   # extract and testing scripts
```

The `extract_all.sh` can also test if the script run properly. 
To do so, pass on a `true` argument to the script. Any other argument (and by default, none) will not test the scripts.

## Solution
The solution is coded with more advanced set-up that haven't been covered in the tutorial. 
For example, it includes logging module, the logging to MLFlow is done along the training. 
Function are split over several files to adopt a modular organization.
**Solutions are updated by hand and not from a script.**

In a similar way, the **fallback script is adapted manually.**

To run the solution, run `uv run solution/main.py`. 
**This script runs all subscripts, logs models to MLFlow, updates data and back-up models in the S3 storage. It doesn't launch a local API.**

To **launch a local API**, run `uvicorn solution.api:app --reload`. You need to have your models stored in MLFlow for it to run properly.

## Contributing

Contributions are welcome! Whether you spotted a bug, a typo, an outdated dependency, or have an idea to improve the tutorial, here's how to get involved:

1. **Open an issue** — head to the [Issues tab](https://github.com/AIML4OS/funathon-project1/issues) and describe what you found or what you'd like to see. Please check that a similar issue doesn't already exist before opening a new one.
2. **Submit a Pull Request** — fork the repository, make your changes on a dedicated branch, and open a PR against `main`. Briefly describe what you changed and why. Linking the PR to the relevant issue is appreciated.

No contribution is too small — fixing a broken link or clarifying a comment is just as valuable as adding a new feature.


## About

This project was developed as part of the **AIML4OS Funathon** — a collaborative hackathon focused on applying AI and machine learning methods to open statistical data.

🔗 [AIML4OS Organization](https://github.com/AIML4OS)

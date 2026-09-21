# Startup Status Classification

## Overview

This project builds a supervised machine-learning system that predicts a startup's current status from company information.

The target column is `status`, with four classes:

- `operating`
- `acquired`
- `closed`
- `ipo`

The project includes exploratory analysis, leakage-aware preprocessing, model comparison, held-out evaluation, model serialization, and a Streamlit frontend.

## Repository Structure

```text
.
├── main.py                         # Streamlit application
├── pyproject.toml                  # Project dependencies and metadata
├── uv.lock                         # Locked dependency versions
├── src/
│   ├── data/companies.csv          # Source dataset
│   ├── models/best_model.pkl       # Selected model artifact
│   └── notebooks/
│       ├── 00_EDA.ipynb
│       ├── 01_Data_Preprocessing.ipynb
│       ├── 02_Model_Training.ipynb
│       ├── 03_Model_Evaluation.ipynb
│       └── 04_Deployment.ipynb
└── README.md
```

## Machine-Learning Workflow

Run the notebooks in this order:

1. `00_EDA.ipynb`: inspect data quality, class balance, missing values, distributions, correlations, and potential outliers.
2. `01_Data_Preprocessing.ipynb`: clean the target, remove identifiers and post-outcome fields, split the data, and define preprocessing.
3. `02_Model_Training.ipynb`: compare logistic regression, decision tree, random forest, and XGBoost using three-fold cross-validation.
4. `03_Model_Evaluation.ipynb`: evaluate the selected model on the untouched test split using per-class metrics and a confusion matrix.
5. `04_Deployment.ipynb`: load the serialized artifact and run a prediction smoke test.

## Data and Feature Policy

The source dataset is `src/data/companies.csv`. The model uses 14 features:

```text
category_code, founded_at, country_code, state_code, city, region,
investment_rounds, invested_companies, funding_rounds, funding_total_usd,
milestones, relationships, lat, lng
```

The following information is excluded from the model:

- identifiers and URLs that do not generalize as business signals
- names and free-text descriptions
- the target column, `status`
- `closed_at` and `ROI`, because they may be known only after the outcome and could leak the target

Numeric features are median-imputed and standardized. Categorical features are filled with their most frequent value and one-hot encoded. These transformations are stored inside the model pipeline.

## Model Selection

The classes are imbalanced, so the training notebook ranks models by macro F1. Macro F1 gives each status class equal importance instead of allowing the majority class to dominate the score.

The current selected model is `random_forest` (`RandomForestClassifier`). The saved artifact contains:

- the preprocessing and classifier pipeline
- the target-label encoder
- the expected feature names
- the selected model name

The artifact is stored at:

```text
src/models/best_model.pkl
```

Keeping these objects together ensures that training-time preprocessing is reused during prediction. A separate scaler file is not required.

## Environment Setup

This project requires Python 3.12 or newer. Install the locked dependencies with:

```bash
uv sync
```

Use the project virtual environment as the kernel when opening the notebooks.

## Run the Streamlit Application

Start the local frontend with:

```bash
uv run streamlit run main.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The application loads `src/models/best_model.pkl`, creates input controls for the stored feature contract, and displays the predicted status and class probabilities.

## Local Demo Reference

The following values come from the first row of the project dataset and can be entered into the Streamlit form:

| Field | Demo value |
| --- | --- |
| Category code | `web` |
| Founded at | `2005-10-17` |
| Country code | `USA` |
| State code | `WA` |
| City | `Seattle` |
| Region | `Seattle` |
| Investment rounds | leave at the default value or use `1` |
| Invested companies | leave at the default value or use `1` |
| Funding rounds | `3` |
| Funding total USD | `39750000` |
| Milestones | `5` |
| Relationships | `17` |
| Latitude | `47.6062095` |
| Longitude | `-122.3320708` |

For this example, the current saved model predicts:

```text
acquired
```

The missing investment values are intentional. The preprocessing pipeline handles missing numeric values with median imputation.

## Validation

Before using the model, review the evaluation notebook rather than relying on accuracy alone. Check:

- macro F1 across all classes
- per-class precision, recall, and F1
- the confusion matrix
- whether the model performs adequately for minority classes

The model is suitable as a project demonstration and should be monitored and revalidated before use in a production decision system.

## To Run

 uv run streamlit run main.py
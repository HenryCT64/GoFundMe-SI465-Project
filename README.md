````md
# Kickstarter Campaign Success Predictor

A machine learning final project that predicts whether a Kickstarter campaign is likely to succeed using only **pre-launch information**.

This project includes:
- a trained **Logistic Regression** model
- a **Streamlit app** for interactive predictions
- extra app pages for **EDA**, **success drivers**, and **model overview**

---

## Project Goal

The goal of this project is to estimate whether a Kickstarter campaign will be **successful**, where success is defined using Kickstarter’s platform outcome:

- `successful` = campaign hit its full funding goal
- `failed` = campaign did not hit its full funding goal

This is a **classification** problem based on historical campaign patterns. The app does **not** determine whether an idea is morally good, creative, or valuable. It only estimates how similar a campaign looks to previously successful Kickstarter projects.

---

## Final Project Structure

```text
/ML-Final-proj
  /Kickstarter_data/      <- 85 CSV snapshot files
  train_model.py          <- full training pipeline
  streamlit_app.py        <- Streamlit app
  model.joblib            <- saved final model
  meta.json               <- saved metadata for app charts/benchmarks
  requirements.txt
  README.md
````

---

## Dataset

The model was trained on **Kickstarter campaign snapshot files** stored in `Kickstarter_data/`.

Pipeline summary:

* load all CSVs
* combine them into one dataset
* deduplicate campaigns by `id`
* keep only campaigns with final states:

  * `successful`
  * `failed`

After cleaning:

* raw rows before deduplication: about **268k**
* rows after deduplication: about **209k**
* finished campaigns used before final cleaning: about **188k**
* cleaned campaigns used for modeling: about **185k**

---

## Final Model

### Algorithm

**Logistic Regression**

This model was chosen because:

* it works well with high-dimensional sparse TF-IDF text features
* it is interpretable
* it clearly outperformed the dummy baseline
* it is more appropriate here than a tree-based model for text-heavy sparse data

---

## Features Used

Only **pre-launch** features were used.

### Text features

* **title** → TF-IDF, unigrams + bigrams
* **blurb** → TF-IDF, unigrams + bigrams

### Numeric features

* **log_goal_usd**
* **campaign_duration**
* **title_len**
* **blurb_len**
* **title_word_count**
* **blurb_word_count**
* **has_blurb**
* **log_goal_per_day**

### Categorical feature

* **main_category**

---

## Best Selected Hyperparameters

The final selected model used:

* **Title TF-IDF max features:** 2500
* **Blurb TF-IDF max features:** 6000
* **min_df:** 5 for both title and blurb
* **Penalty:** L2
* **C:** 2.0
* **Threshold used in final app:** 0.50

Threshold tuning was tested, but the tuned threshold was **not selected** because it reduced macro F1 and made the model worse at identifying failed campaigns.

---

## Model Performance

### Dummy baseline

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 0.6299 |
| Macro F1 | 0.3864 |

### Final selected model

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 0.7370 |
| Macro F1 | 0.7258 |

### Threshold-tuned version (evaluated, not selected)

| Metric   |  Score |
| -------- | -----: |
| Accuracy | 0.7435 |
| Macro F1 | 0.6898 |

### Cross-validation

| Metric           |  Score |
| ---------------- | -----: |
| Best CV Macro F1 | 0.7220 |

---

## Main Findings

Some of the strongest patterns in the data were:

* **Lower funding goals** are strongly associated with success
* **Category matters**: some categories historically succeed much more often than others
* **Campaign duration matters somewhat**, with very long campaigns tending to perform worse
* **Text matters**, especially recurring words/themes in titles and blurbs

Examples of words associated more with success in the final model:

* `photobook`
* `documentary`
* `book`
* `children book`
* `illustrated`
* `comedy`

Examples of words associated more with failure in the final model:

* `want`
* `platform`
* `looking`
* `trying`
* `website`
* `business`

These are correlations learned from the training data, not guarantees.

---

## Setup

It is recommended to use a virtual environment.

### 1. Create and activate a virtual environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Train the Model

Run:

```bash
python train_model.py
```

This script will:

1. load and deduplicate all Kickstarter CSV files
2. engineer text, numeric, and category features
3. split the data into train / validation / test sets
4. run a dummy baseline
5. run GridSearchCV to tune the Logistic Regression pipeline
6. compare the default threshold with a tuned threshold
7. retrain the final selected model
8. save:

   * `model.joblib`
   * `meta.json`

### Expected training time

Training time depends on your laptop, but because the script runs a grid search over multiple combinations, it may take **several minutes to around 15–30 minutes** on a standard machine.

---

## How to Run the App

Run:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

---

## App Pages

The Streamlit app includes four pages:

### 1. Predictor

Enter:

* campaign title
* blurb
* category
* funding goal
* campaign duration

The app returns:

* predicted success probability
* likely outcome
* basic interpretation of the setup
* what-if analysis for changing goal and duration

### 2. Data Explorer

Shows:

* overall success rate
* category success rates
* funding-goal benchmarks
* simple EDA-style summaries

### 3. Success Drivers

Shows:

* major patterns associated with success
* words/features that leaned positive or negative
* strongest and weakest-performing categories

### 4. Model Overview

Shows:

* model type
* features used
* selected hyperparameters
* final performance metrics
* limitations

---

## Example Input

**Title:** `The Art of Urban Sketching`
**Description:** `A coffee-table photobook celebrating street artists around the world.`
**Category:** `Art`
**Goal:** `$5,000`
**Duration:** `30 days`

Possible output:

* success probability around the mid/high 70% range
* likely to succeed
* strong or moderate signal depending on exact model behavior

---

## Limitations

This model has several important limitations:

* it does **not** know the creator’s audience, reputation, or marketing reach
* it does **not** measure idea quality directly
* it only reflects patterns in the historical Kickstarter data available here
* it should be treated as a **decision-support tool**, not a guarantee

---

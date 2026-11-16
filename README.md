# Lead Conversion Prediction

A machine learning project that predicts whether a sales lead will convert into a customer, so the sales team can prioritise follow-ups.

Built as a single Google Colab notebook (`Lead_Conversion_Colab.ipynb`) using Python and scikit-learn (plus XGBoost when available).

## What it does

1. Cleans the lead data and builds features.
2. Explores which factors relate to conversion.
3. Trains four classifiers: Logistic Regression, Decision Tree, Random Forest and XGBoost.
4. Evaluates them with Accuracy, Precision, Recall, F1 and ROC-AUC.
5. Predicts conversion for a new lead and ranks all open leads as HIGH / MEDIUM / LOW priority.

## Dataset

`Unified_Leads.csv`: 27,375 leads (June 2025 to February 2026) with source, campaign, stage, assigned executive and call details. Names, phone numbers and emails are anonymised.

| Item | Value |
|---|---|
| Total leads | 27,375 |
| Converted | 845 |
| Leads with a known outcome (CONVERTED or LOST), used for training | 12,543 (6.7% converted) |
| Open leads (Prospect / Follow Up Required), scored but not trained on | 14,832 |

## Quick start (Google Colab)

1. Go to [colab.research.google.com](https://colab.research.google.com) and choose **File > Upload notebook**.
2. Select `Lead_Conversion_Colab.ipynb`.
3. Choose **Runtime > Run all**.
4. When the first cell shows **Choose files**, upload `Unified_Leads.csv`.
5. At the end, `scored_open_leads.csv` and `lead_model.joblib` download automatically.

The full run takes a few minutes.

## Requirements

Preinstalled in Colab: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `joblib`.

To run locally, install them with `pip install pandas numpy matplotlib scikit-learn xgboost joblib` and remove the `google.colab` upload/download lines. If XGBoost is missing, the notebook uses scikit-learn's Gradient Boosting instead.

## Notebook structure

| Cell | Purpose |
|---|---|
| 1 | Imports and file upload |
| 2 | Load data, view missing values |
| 3 | Cleaning |
| 4 | Define target, separate resolved and open leads, remove leakage |
| 5 | Feature engineering and encoding |
| 6-7 | Exploratory data analysis |
| 8 | Train the four models |
| 9 | Metrics table (Accuracy, Precision, Recall, F1, ROC-AUC) |
| 10 | ROC curve, confusion matrix, lift, feature importance |
| 11 | Check on newer leads (time-based validation) |
| 12 | Predict a single new lead |
| 13 | Score and rank open leads, save outputs |

## Results summary

On a random 20% test split, all four models reach about **0.84-0.85 ROC-AUC**, F1 about 0.62 and accuracy about 95%. Always predicting "no conversion" already scores about 93%, so accuracy alone is not informative.

**Important caveat:** when the models are trained on leads before October 2025 and tested on later leads, ROC-AUC falls to about **0.53-0.59**. Most of the strong signal comes from an early period (one campaign and one executive) with much higher conversion, which faded after September. Treat the model as a rough ranking tool, not a reliable predictor of recent leads. See `DOCUMENTATION.md` for details.

## Outputs

- `scored_open_leads.csv`: open leads ranked by score with a priority label.
- `lead_model.joblib`: saved model with its threshold and settings.

## Limitations

- Scores are rankings, not calibrated probabilities (the models use class weights).
- Features are limited (no call outcomes, lead background or course interest).
- Only about 845 positive examples, and the conversion rate changed sharply over time.

## Repository files

- `Lead_Conversion_Colab.ipynb`: the full project notebook
- `README.md`: this file
- `DOCUMENTATION.md`: detailed technical documentation

## Data privacy

The dataset contains anonymised personal details. Check that you are allowed to publish it before adding `Unified_Leads.csv` to a public repository. If not, leave it out and tell users to supply their own copy.

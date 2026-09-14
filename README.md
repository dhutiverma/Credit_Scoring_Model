# CodeAlpha Task 1 — Credit Scoring Model

## Objective
Predict whether an applicant is creditworthy using historical financial information.

## Approach
A Logistic Regression classification pipeline is used with:
- Median imputation
- Standard scaling
- Logistic Regression
- Train/test split with stratification

## Features
Age, annual income, total debt, credit history, payment history score,
late payments, existing loans, employment years, credit utilization and debt-to-income ratio.

## Evaluation
The project reports:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion matrix
- ROC curve

## Dataset
`data/credit_scoring_dataset.csv` is a reproducible educational dataset generated for this internship project.
Random seed: 42.

## How to run in VS Code

```bash
pip install -r requirements.txt
python credit_scoring_model.py
```

The trained model is saved in `model/` and evaluation files are saved in `outputs/`.

## Repository name
`CodeAlpha_Credit_Scoring_Model`

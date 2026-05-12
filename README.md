# Telecom Customer Churn Prediction (Colab‑only version)

This project predicts customer churn for a telecom operator.  
The code is written for **Google Colab** and loads the dataset from **Google Drive** (link below).

**Important:** The code is not adapted for local execution — it runs only in Colab.  
If you want to run it elsewhere, you will need to modify the file paths or copy the notebook into Colab.

---

## Dataset

The dataset is taken from Kaggle:  
🔗 [Google Drive link to the file](https://drive.google.com/file/d/1q-GB1gHxhyOnkXC_wodLdT4R2BRJ3pTO/view?usp=sharing)

The file must be uploaded to your Google Drive in the following folder:  
`/content/drive/MyDrive/`  
File name: `telecom_users.csv`

---

## How to run (minimum effort)

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook or upload my `.ipynb` file (or copy the `.py` code into a cell).
3. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
Place the file `telecom_users.csv` in your `MyDrive` folder (use the link above to download it).

Run all cells — the code will load the data, preprocess it, and train models.

> ⚠️ If you are using my `.py` file in Colab, just copy the code into a cell or upload it as a notebook.

## What the code does

The code implements a standard classification pipeline:

- Load data and initial exploration
- Handle missing values (TotalCharges, numeric columns)
- Encode categorical features (LabelEncoder, binary mapping)
- Create new features (payment ratios, service counts, flags)
- Scale features (StandardScaler)
- Train 4 models (RandomForest, LogisticRegression, XGBoost, GradientBoosting)
- Evaluate using Accuracy, Precision, Recall, F1, ROC‑AUC
- Visualize (confusion matrix, ROC curve, feature importance)
- Save the best model and results to Google Drive

## Requirements (no installation needed in Colab, but listed for reference)

Libraries used (pre‑installed in Colab):

- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, joblib

If something is missing, run `!pip install <name>`.

## Example results

After running, you will see a table with metrics for all models.  
Typically the best model is XGBoost or RandomForest with ROC‑AUC around **0.82–0.85** on the test set.

## Author

Dmitrii Trofimchuk  
GitHub: [https://github.com/artrokza-lang]

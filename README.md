# Financial Fraud Detection

A Python project that compares fraud detection models and shows their results in a Streamlit dashboard. It uses the three provided datasets separately.

## Run the project

Use **Python 3.12 (64-bit)**. Extract the ZIP, open the `Financial_Fraud_Detection` folder in VS Code, and open a terminal there.

On Windows PowerShell, run these commands one at a time:

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically. Keep the terminal open while using the app. Stop it with Ctrl+C.

Next time, run only the last command. There is no need to activate the environment or install packages again. If a step fails, fix that error before continuing. If Python 3.12 is not found, install its 64-bit Windows version first.

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Files to understand first

| File | What it does |
|---|---|
| `app.py` | Dashboard, filters, model results and transaction predictions |
| `train.py` | Splits the data, trains models and evaluates them |
| `predict.py` | Scores a CSV from the terminal |
| `src/data.py` | Reads the datasets and prepares permitted features |
| `src/models.py` | Defines the models and preprocessing |
| `Financial_Fraud_Detection.ipynb` | Walkthrough with executed outputs |
| `reports/PROJECT_REPORT.md` | Method, findings and limitations |
| `data/` | The three supplied datasets |
| `artifacts/` | Trained models, metrics, split records and cleaned SQLite tables |
| `examples/` | Small CSV inputs to try in the dashboard |

## Try the dashboard

Choose **Bank transactions** first. In Data overview, filter by location, type or date. Model results compares the algorithms. Predict accepts the matching example CSV or a single edited row. Transaction replay processes saved test rows in small batches. About the project explains the evaluation and its limits.

All saved models are included, so you can open the dashboard without retraining.

## Models included

Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting and a deep neural network are the classifiers. Linear Regression, degree-2 Polynomial Regression and SVR are experimental numeric-score baselines. Isolation Forest, One-Class SVM and an autoencoder provide anomaly scores. A dummy model provides a comparison baseline.

Fraud is a classification problem. Regression/anomaly scores are not probabilities. The primary dashboard model is selected from the non-dummy classifiers using validation average precision. The supervised neural network has hidden layers of 64, 32 and 16 units and runs on CPU with scikit-learn.

## Train and evaluate again (optional)

Close the dashboard, then run:

```powershell
& ".\.venv\Scripts\python.exe" train.py --dataset all
& ".\.venv\Scripts\python.exe" scripts/build_report.py
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

These commands replace generated model artifacts, refresh the notebook/report and check split isolation and saved predictions. Training uses up to 20,000 training rows per model by default; kernel models have a 5,000-row limit. Exact counts and warnings are exported. To use all training rows for non-kernel models, add `--max-train 0` to the training command.

## Predict from a CSV (optional)

```powershell
& ".\.venv\Scripts\python.exe" predict.py --dataset bank --input examples/bank_input.csv --output predictions.csv
```

Use the correct template for the selected dataset. Labels are not required. Invalid dates, missing required columns and invalid numeric values are rejected; missing numeric values use training-fitted imputation.

## What the results mean

The bank model caught 8 of 13 test frauds, with 2 false alerts: precision 80%, recall 61.54%. There are few positive examples, so this estimate is uncertain.

The synthetic and credit-card sources show near-baseline ranking performance with the conservative feature set. Their models are included for comparison, not as reliable operational detectors. High accuracy alone would hide poor fraud recall.

All sources are split before learning preprocessing. Bank and credit-card use stratified random splits; synthetic data uses disjoint user groups. These benchmarks do not establish future-time performance. Model scores are uncalibrated. Replay is a simulation; live bank integration, Kafka/Spark, external alerts and online retraining are not implemented.

## Hindi instructions

ZIP extract करें → project folder VS Code में खोलें → ऊपर के तीन Windows commands एक-एक करके चलाएँ। अगले दिन सिर्फ आखिरी command चलानी है। किसी `.bat` file की जरूरत नहीं है। Code समझने के लिए पहले `src/data.py`, फिर `src/models.py`, `train.py` और आखिर में `app.py` पढ़ें।

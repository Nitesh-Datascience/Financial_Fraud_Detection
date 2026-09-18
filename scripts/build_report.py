"""Rebuild measured report, comparison figure and executed explanatory notebook."""

from pathlib import Path
import json, sys, io, contextlib, base64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import CONFIG, load_data

R = ROOT / "reports"
R.mkdir(exist_ok=True)
summaries = []
for key in CONFIG:
    p = ROOT / "artifacts" / key
    a = json.loads((p / "audit.json").read_text())
    m = pd.read_csv(p / "model_metrics.csv")
    best = m[m.model.eq(a["selected_model"])].iloc[0]
    test = pd.read_csv(p / "test_predictions.csv")
    summaries.append(
        {
            "dataset": key,
            "rows": a["rows"],
            "frauds": a["fraud_count"],
            "model": a["selected_model"],
            "test_rows": len(test),
            "test_frauds": int(test.label.sum()),
            "test_AP": best.test_average_precision,
            "AP_baseline": test.label.mean(),
            "precision": best.test_precision,
            "recall": best.test_recall,
            "TP": int(best.test_tp),
            "FP": int(best.test_fp),
            "FN": int(best.test_fn),
            "TN": int(best.test_tn),
            "threshold": best.threshold,
        }
    )
summary = pd.DataFrame(summaries)
summary.to_csv(R / "results_summary.csv", index=False)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axs = plt.subplots(1, 3, figsize=(14, 5), facecolor="#f6f8fc")
for ax, row in zip(axs, summaries):
    ax.set_facecolor("#f6f8fc")
    bars = ax.bar(
        ["Model AP", "Prevalence\nbaseline"],
        [row["test_AP"], row["AP_baseline"]],
        color=["#126b70", "#a9b5c8"],
        width=0.55,
    )
    ax.set_ylim(0, 1)
    ax.set_title(row["dataset"].replace("_", " ").title(), fontweight="bold", pad=18)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.022,
            f"{bar.get_height():.4f}",
            ha="center",
        )
    ax.set_ylabel("Average precision")
    ax.text(
        0.5,
        -0.25,
        f"{row['model']}\nTest: {row['test_rows']:,} rows / {row['test_frauds']:,} frauds",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
fig.suptitle(
    "Measured performance • untouched test partitions",
    fontsize=17,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()
fig.savefig(R / "evaluation.png", dpi=160, bbox_inches="tight")
plt.close(fig)
lines = [
    "# Financial Fraud Detection — Project Report",
    "",
    "## Objective",
    "Identify transactions requiring investigation and compare machine learning, deep learning, regression and anomaly approaches using the supplied data. Streamlit is the implemented dashboard.",
    "",
    "## Data audit",
    "| Source | Reviewed rows | Fraud rows | Excluded labels |",
    "|---|---:|---:|---:|",
]
for k in CONFIG:
    a = json.loads((ROOT / "artifacts" / k / "audit.json").read_text())
    lines.append(
        f"| {k} | {a['rows']:,} | {a['fraud_count']:,} | {a['excluded_unreviewed']} |"
    )
lines += [
    "",
    "The bank workbook also contains `isFraud - Copy`, a duplicate target representation, and two `Not reviewed` records. Neither label copy is a feature; unreviewed rows are not assigned a negative label. The credit-card CSV is not the anonymized V1–V28 dataset assumed in some reference notebooks.",
    "",
    "## Method",
    "1. Load each source independently; remove exact duplicate records and reject unresolved labels.",
    "2. Split before fitting any transformation. Bank and credit-card use stratified 60/20/20 splits; synthetic uses disjoint users with approximately the same fractions.",
    "3. Use explicit feature allowlists, calendar extraction and log amount; impute/scale numerics and encode categoricals using training data only.",
    "4. Train 13 candidate models. Fixed architecture and hyperparameters, with at most 20,000 stratified training rows and 5,000 for expensive kernel models. No evaluation rows are undersampled.",
    "5. Optimize a separate threshold for each model using validation F2. Choose the primary classifier by validation AP, then evaluate frozen choices on test data.",
    "6. Persist a selected pipeline per source, audited row splits, metrics and test predictions. Display test data in the dashboard; use the same pipeline for uploaded data.",
    "",
    "## Measured results",
    "| Dataset | Selected classifier | Test AP | Baseline AP | Precision | Recall | TP / FP / FN / TN |",
    "|---|---|---:|---:|---:|---:|---|",
]
for r in summaries:
    lines.append(
        f"| {r['dataset']} | {r['model']} | {r['test_AP']:.4f} | {r['AP_baseline']:.4f} | {r['precision']:.2%} | {r['recall']:.2%} | {r['TP']} / {r['FP']} / {r['FN']} / {r['TN']} |"
    )
lines += [
    "",
    "![Measured average precision](evaluation.png)",
    "",
    "The bank model detects 8 of 13 test frauds and produces 2 false alerts. Its performance is promising within this benchmark, but 13 positives give high sampling uncertainty. Random splits can also overstate future-time performance.",
    "",
    "Synthetic and credit-card models have near-baseline ranking performance under these conservative features. The synthetic model flags all test rows at its validation F2 threshold; this is not useful operational detection. Credit-card precision is about 1%, close to its 1% prevalence. High default-label accuracy (99% by predicting no credit-card fraud) would disguise failure to detect fraud. No claim is made that these labels are random: feature inadequacy, generation logic and label quality require investigation.",
    "",
    "## Feature availability decisions",
    "Bank: transaction amount, old sender/recipient balances, type, branch, account type, time-of-day and supplied calendar features. Old balances are assumed available pre-event. Exclude new balances, IDs, flags, duplicated labels, arbitrary row counters and ambiguous unusual-login values.",
    "",
    "Synthetic: amount, card age, transaction/device/merchant/card categories, location and calendar. Exclude user/transaction IDs and ambiguous account-balance, prior-fraud and whole-day transaction-count fields until their timing and provenance are confirmed. User IDs are used only for partitioning.",
    "",
    "Credit-card: amount, transaction type, location and calendar. Transaction and merchant IDs are excluded from this conservative benchmark. More informative merchant/customer history may improve results if computed strictly from past events.",
    "",
    "## Reference notebook review",
    "- `Advanced_Financial_Fraud_Detection_Model_With_Dashboard (1)(2).ipynb`: useful transaction-analysis ideas, but hardcoded local paths, inconsistent labels and references to undefined model/test variables prevent a clean end-to-end run. The rebuilt pipeline uses the actual workbook schema.",
    "- `Financial_Fraud_Detection_Cleaned(2).ipynb` and `Fraud_Detection_(1)(2).ipynb`: expect `creditcard.csv` with `Class` and/or V1–V28 features, unlike the supplied credit-card CSV. Some transformations/undersampling occur before later evaluation; the rebuilt project separates partitions before fitting.",
    "- `EDA(1).ipynb`: exploratory reference with a generic `data.csv` path; the rebuilt loaders use explicit source names.",
    "- `Data_Visualization_Info(2).ipynb`: generic visualization examples, not a complete fraud dashboard.",
    "- `Data_Augmentation(2).ipynb`: image, text and generic time-series augmentation examples; these are not appropriate transaction-label augmentation and are not applied.",
    "",
    "## Dashboard features",
    "The dashboard uses five simple tabs: Data overview, Model results, Predict, Transaction replay and About the project. Overview contains location/type/date filters, label and alert counts, trends, score distribution, a confusion matrix and a downloadable review queue. Model results includes evaluation metrics and a precision-recall curve. Prediction supports CSV batches and one edited transaction. Replay processes 25 held-out rows per click. The About tab explains source checks, exclusions and warnings.",
    "",
    "## Implemented versus extension scope",
    "Implemented: local SQLite ETL, ML/deep learning training, anomaly comparison, inference, test reporting and Streamlit dashboard. Kafka/Spark ingestion, graph-based network detection, external email/SMS alerts and adaptive online retraining are future extensions from the broad brief. No production bank integration or Power BI file is claimed.",
    "",
    "## Validation and reproducibility",
    "The included tests verify non-overlapping splits, user isolation, target exclusion, schema rejection, unknown-category handling, all 13 metric records, confusion totals and serialized-model prediction consistency. Streamlit AppTest exercises every dataset, single prediction, replay and threshold changes. Package versions and random seeds are recorded; training logs and model-specific warnings are included.",
    "",
    "## Presentation / viva notes",
    "- Why classification? Fraud is a binary label; regression models are comparative numeric-score baselines.",
    "- Why not accuracy alone? A majority-only classifier reaches 99% accuracy on the supplied credit-card dataset while detecting zero fraud.",
    "- Why validation and test? Validation chooses the model and threshold; the test set measures those frozen decisions.",
    "- What is deep learning here? A feed-forward MLP with three hidden layers, plus an encoder/bottleneck/decoder reconstruction model.",
    "- Why no image augmentation or blanket outlier removal? Transaction labels need domain-valid transformations; extreme transactions may be the fraud signal.",
    "- What is the strongest limitation? Data provenance and feature timing are unverified, and two datasets show weak signal. Bank positives are scarce.",
    "- What is real-time here? Local replay and interactive prediction only; real streaming infrastructure is an extension.",
    "",
    "## References",
    "[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) · [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) · [Streamlit testing](https://docs.streamlit.io/develop/api-reference/app-testing)",
]
(R / "PROJECT_REPORT.md").write_text("\n".join(lines) + "\n")
# Build notebook with actually evaluated cells and captured outputs, not fabricated runs.
cells = []
env = {}
count = 0


def md(text):
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
    )


def code(text):
    global count
    count += 1
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(text, "notebook_cell", "exec"), env)
    outputs = []
    if out.getvalue():
        outputs = [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": out.getvalue().splitlines(True),
            }
        ]
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": count,
            "source": text.splitlines(True),
            "outputs": outputs,
        }
    )


md(
    "# Financial Fraud Detection Model\nAn executed, reproducible walkthrough of the supplied datasets and trained results. Run from the extracted project folder. See `train.py` and `src/` for the full reusable implementation.\n\n**Scope:** local academic prototype. Models on two sources perform near baseline; no production accuracy claim is made."
)
code(
    "from pathlib import Path\nimport sys, json\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    raise RuntimeError('Open this notebook from Financial_Fraud_Detection')\nsys.path.insert(0, str(ROOT))\nimport pandas as pd, numpy as np, joblib\nfrom src.data import CONFIG, load_data, features\nfrom src.models import score\nprint('Datasets:', ', '.join(CONFIG))"
)
md(
    "## 1. Source audit\nThe datasets have distinct schemas and are modeled independently. Unreviewed bank rows are excluded, not marked safe."
)
code(
    "audits=[]\nfor key in CONFIG:\n    df, audit=load_data(key)\n    audits.append({'dataset':key, **audit})\nprint(pd.DataFrame(audits).to_string(index=False))"
)
md(
    "## 2. Explicit features\nOnly allowlisted, assumed pre-event inputs are used. Labels and label copies cannot leak into X. Calendar/log features are deterministic; learned preprocessing fits training data only."
)
code(
    "for key in CONFIG:\n    df,_=load_data(key)\n    x=features(df,key)\n    print(key, ':', ', '.join(x.columns))"
)
md(
    "## 3. Split evidence\nSynthetic transactions are partitioned by user; other sources use stratified random splits. Random benchmarks are not a substitute for future-time evaluation."
)
code(
    "for key in CONFIG:\n    p=ROOT/'artifacts'/key\n    audit=json.loads((p/'audit.json').read_text())\n    split=pd.read_csv(p/'split_manifest.csv')\n    print(key, audit['split_method'])\n    print(split.groupby('split').agg(rows=('row_id','size'), fitted_rows=('used_for_fit','sum')).to_string())"
)
md(
    "## 4. Algorithms and training\nThe full run includes six classifiers, a dummy baseline, three experimental regressors and three anomaly methods. The neural classifier uses 64→32→16 hidden units; the reconstruction autoencoder uses 32→8→32. Regressor/anomaly scores are not probabilities.\n\nTo retrain from this folder, run `python train.py --dataset all` in a terminal, then `python scripts/build_report.py`. Training is explicit so reopening this notebook does not silently replace saved models."
)
code(
    "from src.models import candidates\nfor name,(kind,model) in candidates().items():\n    print(name, '|', kind, '|', model)"
)
md(
    "## 5. Validation-selected model comparison\nPrimary classifier selection uses validation average precision. Thresholds maximize validation F2. Test rows are used only after these choices are frozen."
)
code(
    "for key in CONFIG:\n    m=pd.read_csv(ROOT/'artifacts'/key/'model_metrics.csv')\n    print('\\n'+key.upper())\n    print(m[['model','fit_rows','val_average_precision','test_average_precision','test_precision','test_recall']].to_string(index=False))"
)
md(
    "## 6. Summarize the measured results\nAverage precision is compared with test class prevalence. Similar values indicate little useful ranking improvement."
)
code(
    "summary=pd.read_csv(ROOT/'reports'/'results_summary.csv')\nprint(summary.to_string(index=False))"
)
# Embed the real generated figure as a notebook display output.
count += 1
cells.append(
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": count,
        "source": [
            "from IPython.display import Image, display\n",
            'display(Image(filename=str(ROOT / "reports" / "evaluation.png")))',
        ],
        "outputs": [
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(
                        (R / "evaluation.png").read_bytes()
                    ).decode(),
                    "text/plain": ["Measured test performance"],
                },
            }
        ],
    }
)
md(
    "## 7. Reload and score held-out transactions\nInference uses the same serialized preprocessing as training. Extra columns, including labels, are not features."
)
code(
    "key='bank'\nb=joblib.load(ROOT/'artifacts'/key/'best_model.joblib')\ntest=pd.read_csv(ROOT/'artifacts'/key/'test_predictions.csv')\nscores=score(b['model'],features(test.head(5),key),b['kind'])\nprint(pd.DataFrame({'risk_score':scores,'flagged':scores>=b['threshold'],'actual':test.label.head(5)}).to_string(index=False))\nnp.testing.assert_allclose(scores,test.risk_score.head(5),rtol=1e-6,atol=1e-9)\nprint('Serialized inference matches stored evaluation.')"
)
md(
    "## 8. SQL ETL inspection\nThese tables contain the cleaned full sources. They are for analysis, not reported out-of-sample model performance."
)
code(
    "import sqlite3\nwith sqlite3.connect(ROOT/'artifacts'/'fraud_detection.db') as con:\n    for key in CONFIG:\n        print(key,pd.read_sql_query(f'SELECT COUNT(*) AS rows, SUM(label) AS frauds FROM {key}_transactions',con).to_dict('records'))"
)
md(
    "## 9. Interpretation and next steps\nBank: 8/13 test frauds detected with 2 false positives. Small positive counts and random splits limit generalization.\n\nSynthetic and credit-card: near-baseline performance. F2 optimization produces excessive alerts; this is documented model weakness, not a successful operational detector.\n\nNext: verify labels and feature timing; obtain past-only behavioral features; run temporal validation; calibrate and select cost-aware thresholds. Kafka/Spark, external alerts, graph models and online adaptation remain extensions.\n\nLaunch dashboard: `python -m streamlit run app.py`."
)
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, cell in enumerate(cells):
    cell["id"] = f"fraud-cell-{i:02d}"
(ROOT / "Financial_Fraud_Detection.ipynb").write_text(json.dumps(nb, indent=1))
print("Report, figure and executed notebook created.")

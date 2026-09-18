"""Run this dashboard with: python -m streamlit run app.py"""

from pathlib import Path
import json
import sys
from html import escape

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import CONFIG, features, load_data
from src.models import score

st.set_page_config(page_title="Financial Fraud Detection", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#6965db", "#ed9478", "#58bba6", "#b9b5ec"]

st.markdown(
    """<style>
.stApp {background:#f5f5fb;color:#292b45;}
[data-testid="stHeader"] {background:rgba(245,245,251,.94);}
.block-container {max-width:1500px;padding:2rem 2.4rem 3rem;}
[data-testid="stSidebar"] {background:#eeedf9;border-right:1px solid #deddf0;}
h1,h2,h3 {color:#282644;letter-spacing:-.04em;}
h3 {font-size:1.2rem !important;}
[data-testid="stCaptionContainer"] {color:#74748d;}
.brand {display:flex;gap:12px;align-items:center;margin:6px 0 30px;}
.brand-icon {padding:12px;background:#6965db;border-radius:14px;color:white;font-weight:800;letter-spacing:-1px;}
.brand-title {font-size:18px;font-weight:750;color:#33304e;line-height:1.35;}
.brand-sub {font-size:10px;letter-spacing:1.7px;color:#85809f;text-transform:uppercase;}
.hero {border-radius:24px;padding:31px 35px;background:linear-gradient(115deg,#28234b,#484076 65%,#6457a0);position:relative;overflow:hidden;color:white;display:flex;justify-content:space-between;align-items:center;gap:24px;margin-bottom:24px;}
.hero:after {content:'';position:absolute;right:-45px;top:-100px;width:290px;height:290px;border:45px solid rgba(213,203,255,.08);border-radius:50%;pointer-events:none;}
.hero-copy {position:relative;z-index:1;}
.hero h1 {color:#fff;font-size:2.2rem;line-height:1.2;margin:10px 0;padding:0;letter-spacing:-1.2px;}
.hero p {color:#d7d2eb;margin:0;font-size:14px;}
.kicker {font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#c6bced;font-weight:700;}
.hero-tag {position:relative;z-index:1;background:#d8f3df;color:#285b44;border-radius:30px;padding:10px 16px;font-size:12px;white-space:nowrap;}
.model-card {background:#fff;border:1px solid #dedcee;border-radius:16px;padding:18px;margin:18px 0;}
.model-card small {color:#87829f;font-size:10px;letter-spacing:1.4px;text-transform:uppercase;}
.model-card strong {display:block;margin-top:8px;color:#514892;font-size:16px;}
.kpi {padding:22px 22px 18px;background:white;border:1px solid #e6e4f1;border-radius:19px;box-shadow:0 5px 16px #3c315005;min-height:154px;margin-bottom:20px;}
.kpi-top {display:flex;justify-content:space-between;align-items:center;color:#75718b;font-size:12px;}
.kpi-icon {padding:6px 9px;border-radius:8px;background:#eeebfd;color:#7260c8;font-size:14px;font-weight:700;}
.kpi-value {font-size:32px;font-weight:750;color:#302a4b;letter-spacing:-1.2px;line-height:1.5;margin-top:5px;}
.kpi-note {font-size:11px;color:#858197;}
.kpi.mint {background:#eaf7f0;border-color:#d7eddf;}.kpi.mint .kpi-icon{background:#d3eedf;color:#3a8161;}
.kpi.peach {background:#fff1eb;border-color:#f5dfd2;}.kpi.peach .kpi-icon{background:#ffe0d1;color:#af6c4c;}
.kpi.lilac {background:#eeecff;border-color:#e1dbfc;}
[data-testid="stMetric"] {padding:20px;background:#fff;border:1px solid #e3e0ee;border-radius:17px;}
[data-testid="stMetricValue"] {color:#4a3d79;font-size:28px;}
[data-testid="stVerticalBlockBorderWrapper"]>div {background:#fff;border-color:#e5e2f0 !important;border-radius:18px !important;}
[data-baseweb="tab-list"] {gap:10px !important;background:#eae8f4;padding:6px;border-radius:13px;margin-bottom:20px;}
[data-baseweb="tab"] {border-radius:9px;padding:10px 18px !important;height:43px;color:#77728b;}
[data-baseweb="tab"][aria-selected="true"] {background:#fff;color:#5b4cb1;box-shadow:0 2px 5px #40306b0c;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"] {display:none;}
.stButton>button,.stDownloadButton>button {border-radius:10px;border-color:#ded9f0;color:#5c4a9c;}
.stButton>button[kind="primary"] {background:#6965db;color:white;border:none;}
[data-testid="stDataFrame"] {border:1px solid #e1dfeb;border-radius:12px;overflow:hidden;}
.outcome-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:15px 0 20px;}
.outcome {padding:17px;border-radius:12px;background:#f1f0fa;}
.outcome strong {display:block;font-size:27px;color:#4d4380;}
.outcome span {font-size:12px;color:#706983;}
.outcome.good {background:#e9f6ef;}.outcome.good strong{color:#3b8565;}
.outcome.miss {background:#fff0e9;}.outcome.miss strong{color:#bc7050;}
.footer {margin-top:30px;padding-top:16px;border-top:1px solid #dfdced;color:#9290a4;font-size:11px;letter-spacing:.5px;}
@media(max-width:800px){.block-container{padding:1.2rem .9rem;}.hero{padding:24px;flex-direction:column;align-items:flex-start;}.hero h1{font-size:1.7rem;}.kpi-value{font-size:26px;}[data-baseweb="tab"]{padding:8px 12px !important;}}
</style>""",
    unsafe_allow_html=True,
)


def summary_card(column, title, value, note, icon, color=""):
    column.markdown(
        f'<div class="kpi {color}"><div class="kpi-top">{escape(title)}'
        f'<span class="kpi-icon">{icon}</span></div>'
        f'<div class="kpi-value">{escape(str(value))}</div>'
        f'<div class="kpi-note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


DATASET_NAMES = {
    "bank": "Bank transactions",
    "synthetic": "Synthetic transactions",
    "credit_card": "Credit-card transactions",
}


@st.cache_resource
def load_model(dataset, modified_time):
    """Keep the fitted model in memory between dashboard interactions."""
    return joblib.load(ROOT / "artifacts" / dataset / "best_model.joblib")


@st.cache_data
def load_results(dataset, modified_time):
    folder = ROOT / "artifacts" / dataset
    predictions = pd.read_csv(folder / "test_predictions.csv")
    comparison = pd.read_csv(folder / "model_metrics.csv")
    audit = json.loads((folder / "audit.json").read_text())
    return predictions, comparison, audit


def draw_chart(figure):
    figure.update_layout(
        margin=dict(l=15, r=15, t=50, b=55),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#7b7690"),
        title=dict(font=dict(size=15, color="#35304f")),
        legend=dict(orientation="h", y=-0.23, x=0, title_text=""),
    )
    figure.update_xaxes(gridcolor="#f0edf6", zeroline=False)
    figure.update_yaxes(gridcolor="#f0edf6", zeroline=False)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def score_transactions(frame, model_info, dataset, threshold):
    """Use the same feature preparation that was used during training."""
    inputs = features(frame, dataset)
    scores = score(model_info["model"], inputs, model_info["kind"])
    result = frame.copy()
    result["risk_score"] = scores
    result["flagged_for_review"] = (scores >= threshold).astype(int)
    return result


st.sidebar.markdown(
    '<div class="brand"><div class="brand-icon">FD</div><div><div class="brand-title">Financial Fraud<br>Detection</div><div class="brand-sub">Analysis workspace</div></div></div>',
    unsafe_allow_html=True,
)

dataset = st.sidebar.selectbox(
    "Dataset", list(CONFIG), format_func=lambda key: DATASET_NAMES[key]
)
settings = CONFIG[dataset]
model_path = ROOT / "artifacts" / dataset / "best_model.joblib"
if not model_path.exists():
    st.error("The trained model is missing. Run: python train.py --dataset all")
    st.stop()

modified_time = model_path.stat().st_mtime
model_info = load_model(dataset, modified_time)
predictions, comparison, audit = load_results(dataset, modified_time)

st.sidebar.markdown(
    f'<div class="model-card"><small>Selected model</small><strong>{escape(model_info["name"])}</strong></div>',
    unsafe_allow_html=True,
)
threshold = st.sidebar.slider(
    "Alert threshold",
    0.0,
    1.0,
    float(model_info["threshold"]),
    0.001,
    key=f"threshold_{dataset}",
)
st.sidebar.caption(
    "Changing this value updates alerts. Saved evaluation results stay the same."
)

st.markdown(
    f'<div class="hero"><div class="hero-copy"><div class="kicker">TRANSACTION ANALYTICS / {escape(DATASET_NAMES[dataset].upper())}</div><h1>A clearer view of transaction risk.</h1><p>Follow the patterns. Compare the models. Review the details.</p></div><div class="hero-tag">● &nbsp; Held-out test data</div></div>',
    unsafe_allow_html=True,
)

summary_tab, models_tab, predict_tab, replay_tab, about_tab = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Check a transaction",
        "Replay",
        "About",
    ]
)

with summary_tab:
    with st.expander("Filter transactions", expanded=False):
        left, right = st.columns(2)
        locations = left.multiselect(
            "Location or branch",
            sorted(predictions[settings["location"]].dropna().unique()),
            key=f"locations_{dataset}",
        )
        transaction_types = right.multiselect(
            "Transaction type",
            sorted(predictions[settings["category"]].dropna().unique()),
            key=f"types_{dataset}",
        )
        filtered = predictions.copy()
        date_format = {
            "bank": None,
            "synthetic": "%d %B %Y",
            "credit_card": "%d-%m-%Y",
        }[dataset]
        filtered["date"] = pd.to_datetime(
            filtered[settings["date"]], format=date_format, errors="coerce"
        )
        if filtered["date"].notna().any():
            date_range = st.date_input(
                "Date range",
                (filtered["date"].min().date(), filtered["date"].max().date()),
                key=f"dates_{dataset}",
            )
            if len(date_range) == 2:
                filtered = filtered[filtered["date"].dt.date.between(*date_range)]
        if locations:
            filtered = filtered[filtered[settings["location"]].isin(locations)]
        if transaction_types:
            filtered = filtered[filtered[settings["category"]].isin(transaction_types)]

    filtered["alert"] = (filtered["risk_score"] >= threshold).astype(int)
    cards = st.columns(4)
    fraud_rate = f"{filtered['label'].mean():.2%}" if len(filtered) else "—"
    summary_card(
        cards[0],
        "Transactions",
        f"{len(filtered):,}",
        "In the current test selection",
        "↗",
    )
    summary_card(
        cards[1],
        "Observed frauds",
        f"{filtered['label'].sum():,}",
        "Confirmed labels in the dataset",
        "◎",
        "mint",
    )
    summary_card(
        cards[2],
        "Review queue",
        f"{filtered['alert'].sum():,}",
        f"Score threshold: {threshold:.3f}",
        "!",
        "peach",
    )
    summary_card(
        cards[3],
        "Fraud rate",
        fraud_rate,
        "Observed frauds / transactions",
        "%",
        "lilac",
    )

    if dataset != "bank":
        st.caption(
            "This model performs near the baseline on test data. Its alerts are not reliable enough for operational use."
        )

    if filtered.empty:
        st.write("No transactions match these filters.")
    else:
        left, right = st.columns(2)
        with left, st.container(border=True):
            daily = (
                filtered.groupby("date")
                .agg(observed_fraud=("label", "sum"), alerts=("alert", "sum"))
                .reset_index()
            )
            draw_chart(
                px.line(
                    daily,
                    x="date",
                    y=["observed_fraud", "alerts"],
                    title="Transaction activity",
                    markers=True,
                    labels={"date": "Date", "value": "Transactions"},
                )
            )
        with right, st.container(border=True):
            by_type = (
                filtered.groupby(settings["category"])["label"].mean().reset_index()
            )
            figure = px.bar(
                by_type,
                x=settings["category"],
                y="label",
                title="Observed fraud rate by transaction type",
                labels={"label": "Fraud rate"},
            )
            figure.update_yaxes(tickformat=".1%")
            draw_chart(figure)

        left, right = st.columns(2)
        with left, st.container(border=True):
            matrix = confusion_matrix(
                filtered["label"], filtered["alert"], labels=[0, 1]
            )
            st.subheader("How the alerts compare")
            st.caption("Actual labels compared with the current alert threshold")
            tn, fp, fn, tp = matrix.ravel()
            st.markdown(
                f'<div class="outcome-grid"><div class="outcome good"><strong>{tp:,}</strong><span>Frauds caught</span></div>'
                f'<div class="outcome miss"><strong>{fn:,}</strong><span>Frauds missed</span></div>'
                f'<div class="outcome"><strong>{fp:,}</strong><span>False alerts</span></div>'
                f'<div class="outcome good"><strong>{tn:,}</strong><span>Correctly not flagged</span></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Counts follow your active filters. An alert is a request for review."
            )
        with right, st.container(border=True):
            log_counts = st.toggle(
                "Logarithmic count scale", value=True, key=f"log_{dataset}"
            )
            draw_chart(
                px.histogram(
                    filtered,
                    x="risk_score",
                    nbins=40,
                    title="Model score distribution",
                    log_y=log_counts,
                    labels={"risk_score": "Model score"},
                )
            )

        st.subheader("Review queue")
        queue = filtered[filtered["alert"].eq(1)].sort_values(
            "risk_score", ascending=False
        )
        queue = queue.drop(columns="date")
        display_columns = [
            settings["date"],
            settings["category"],
            settings["location"],
            settings["amount"],
            "risk_score",
            "label",
        ]
        display_queue = queue[display_columns].copy()
        display_queue["label"] = display_queue["label"].map(
            {0: "Legitimate", 1: "Fraud"}
        )
        st.dataframe(
            display_queue.head(500),
            hide_index=True,
            width="stretch",
            column_config={
                settings["date"]: "Date",
                settings["category"]: "Type",
                settings["location"]: "Location",
                settings["amount"]: st.column_config.NumberColumn(
                    "Amount", format="%.2f"
                ),
                "risk_score": st.column_config.ProgressColumn(
                    "Model score", min_value=0, max_value=1, format="%.3f"
                ),
                "label": "Actual label",
            },
        )
        st.caption(
            "The table shows up to 500 rows. The download contains the full filtered queue."
        )
        st.download_button(
            "Download review queue",
            queue.to_csv(index=False),
            "review_queue.csv",
            "text/csv",
        )

with models_tab:
    st.subheader("Compare the models")
    st.caption(
        "The primary classifier was selected using validation average precision, not test accuracy."
    )
    selected = comparison[comparison["model"].eq(model_info["name"])].iloc[0]
    columns = st.columns(3)
    columns[0].metric("Test precision", f"{selected['test_precision']:.1%}")
    columns[1].metric("Test recall", f"{selected['test_recall']:.1%}")
    columns[2].metric(
        "Test average precision", f"{selected['test_average_precision']:.3f}"
    )

    figure = px.bar(
        comparison,
        x="test_average_precision",
        y="model",
        orientation="h",
        title="Test average precision",
        labels={"test_average_precision": "Average precision"},
    )
    figure.add_vline(x=predictions["label"].mean(), line_dash="dash")
    figure.update_layout(height=520)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "The dashed line is test fraud prevalence: a useful baseline for average precision."
    )
    fields = [
        "model",
        "kind",
        "fit_rows",
        "val_average_precision",
        "test_average_precision",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "threshold",
    ]
    st.dataframe(comparison[fields], hide_index=True, width="stretch")
    st.download_button(
        "Download comparison",
        comparison.to_csv(index=False),
        "model_metrics.csv",
        "text/csv",
    )
    precision, recall, _ = precision_recall_curve(
        predictions["label"], predictions["risk_score"]
    )
    draw_chart(
        px.line(
            x=recall,
            y=precision,
            labels={"x": "Recall", "y": "Precision"},
            title="Selected model: precision-recall curve",
        )
    )
    st.caption(
        "Regression and anomaly scores are not probabilities. Thresholds were chosen on validation data using F2, which favors recall."
    )

with predict_tab:
    st.subheader("Check transactions")
    st.write(
        "Download the template for your selected dataset, fill in your rows and upload it here."
    )
    required = settings["numeric"] + settings["categorical"] + [settings["date"]]
    source, _ = load_data(dataset)
    template = source[required].head(5)
    st.download_button(
        "Download input template",
        template.to_csv(index=False),
        f"{dataset}_input.csv",
        "text/csv",
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv", key=f"upload_{dataset}"
    )
    if uploaded_file is not None:
        try:
            incoming = pd.read_csv(uploaded_file)
            if not 0 < len(incoming) <= 100000:
                raise ValueError("Please upload between 1 and 100,000 rows.")
            result = score_transactions(incoming, model_info, dataset, threshold)
            st.write(
                f"Scored {len(result):,} transactions; {result['flagged_for_review'].sum():,} flagged."
            )
            st.dataframe(result.head(500), hide_index=True, width="stretch")
            st.download_button(
                "Download predictions",
                result.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
        except (ValueError, TypeError, KeyError, pd.errors.ParserError) as error:
            st.error(str(error))

    st.subheader("Try one transaction")
    edited = st.data_editor(template.head(1), hide_index=True, key=f"single_{dataset}")
    if st.button("Predict", type="primary"):
        try:
            result = score_transactions(edited, model_info, dataset, threshold).iloc[0]
            st.metric("Model score", f"{result['risk_score']:.4f}")
            if result["flagged_for_review"]:
                st.warning("Flagged for review")
            else:
                st.success("Below the current alert threshold")
            st.caption(
                "This score is not a calibrated probability or confirmation of fraud."
            )
        except (ValueError, TypeError, KeyError) as error:
            st.error(str(error))

with replay_tab:
    st.subheader("Transaction replay")
    st.write(
        "This is a local simulation using test transactions, not a live bank feed."
    )
    state_key = f"replay_{dataset}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    left, right = st.columns(2)
    if left.button("Next 25 transactions"):
        st.session_state[state_key] = min(
            len(predictions), st.session_state[state_key] + 25
        )
    if right.button("Reset"):
        st.session_state[state_key] = 0
    processed = st.session_state[state_key]
    batch = predictions.head(processed).copy()
    batch["alert"] = (batch["risk_score"] >= threshold).astype(int)
    st.progress(
        processed / len(predictions),
        text=f"{processed:,} of {len(predictions):,} transactions",
    )
    st.write(f"Alerts generated: {batch['alert'].sum():,}")
    if processed:
        st.dataframe(batch.tail(25), hide_index=True, width="stretch")

with about_tab:
    st.subheader("How the project works")
    st.write(
        "Each dataset is cleaned and modeled separately. Preprocessing is fitted on training rows. Validation data selects the classifier and threshold; the test set measures the frozen choices."
    )
    st.write("**Split:** " + audit["split_method"])
    st.write(
        "**Models:** Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, a three-hidden-layer neural network, three regression baselines, three anomaly models and a dummy baseline."
    )
    st.write(
        "**Excluded inputs:** Labels and label copies, identifiers, fraud flags, post-transaction balances and behavior fields whose timing is unclear. Bank old balances are assumed available before the transaction."
    )
    st.write(
        "**Limits:** Bank evaluation has only 13 test frauds. The other two sources have near-baseline results. Scores are not calibrated probabilities, currencies are unspecified, and these results do not establish production readiness."
    )
    st.write(
        "The default training limit is 20,000 rows per model, with a 5,000-row limit for kernel methods. Validation and test sets are not undersampled. Kafka, external notifications and online retraining are not included."
    )
    with st.expander("Data audit"):
        st.json(audit)
    warnings = comparison.loc[
        comparison["warnings"].fillna("").ne(""), ["model", "warnings"]
    ]
    if not warnings.empty:
        with st.expander("Training notes"):
            st.dataframe(warnings, hide_index=True)

st.markdown(
    '<div class="footer">FINANCIAL FRAUD DETECTION &nbsp; / &nbsp; Explore · Compare · Review</div>',
    unsafe_allow_html=True,
)

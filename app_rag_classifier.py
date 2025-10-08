# app_rag_classifier.py
# Ticket RAG + Urgency Classifier (aligned with your notebook cleaning)

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import load as joblib_load, dump as joblib_dump

st.set_page_config(page_title="Ticket RAG + Urgency Classifier", layout="wide")

# ----------------------------- Utilities -----------------------------

def clean_text_keep_case(s: str) -> str:
    """Notebook-aligned cleaner: keep case & apostrophes, trim, collapse spaces."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9'\s]", " ", s)
    return s

def expand_product_placeholders_series(desc: pd.Series, prod: pd.Series | None) -> pd.Series:
    """Replace {product_purchased} (any casing/underscore/space) with Product Purchased."""
    if prod is None:
        return desc.fillna("").astype(str)
    rx = re.compile(r"\{product[_ ]purchased\}", flags=re.IGNORECASE)
    out = []
    for d, p in zip(desc.fillna("").astype(str), prod.fillna("").astype(str)):
        out.append(rx.sub(p, d))
    return pd.Series(out, index=desc.index)

def build_text_clean(df: pd.DataFrame) -> pd.Series:
    """Use df['text_clean'] if present; else expand placeholders, then clean (keep case)."""
    if "text_clean" in df.columns:
        return df["text_clean"].astype(str).fillna("").apply(clean_text_keep_case)

    desc = df.get("Ticket Description", pd.Series([""] * len(df), index=df.index)).astype(str)
    prod = df.get("Product Purchased", None)
    desc_expanded = expand_product_placeholders_series(desc, prod)
    subj = df.get("Ticket Subject", pd.Series([""] * len(df), index=df.index)).astype(str)
    return (subj.fillna("") + " " + desc_expanded.fillna("")).apply(clean_text_keep_case)

def extractive_summary(query: str, docs: list[str], max_sentences: int = 3) -> str:
    """Simple extractive summary via TF-IDF sentence scoring with redundancy penalty."""
    text = " ".join([str(d) for d in docs])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "No summary available."

    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    S = vec.fit_transform(sentences)
    q = vec.transform([query])

    base = linear_kernel(q, S).ravel()

    chosen_idx = []
    for _ in range(min(max_sentences, len(sentences))):
        best_i, best_score = None, -1e9
        for i in range(len(sentences)):
            if i in chosen_idx:
                continue
            redundancy = 0.0 if not chosen_idx else max(linear_kernel(S[i], S[chosen_idx]).ravel())
            score = base[i] - 0.4 * redundancy
            if score > best_score:
                best_i, best_score = i, score
        chosen_idx.append(best_i)

    return " ".join([sentences[i] for i in chosen_idx])

# ---------- Robust prediction for text-only OR column-transformer pipelines ----------

def predict_proba_safe(model, text: str) -> float:
    """
    Works with:
      A) text-only pipeline: Pipeline([('tfidf', ...), ('clf', CalibratedClassifierCV)])
      B) column-transformer pipeline: Pipeline([('feats', ColumnTransformer), ('clf', ...)])
         expecting a DataFrame that contains at least a 'text_all' column.
    """
    # Text-only pipeline?
    if hasattr(model, "named_steps") and "tfidf" in model.named_steps:
        return float(model.predict_proba([text])[:, 1])

    # ColumnTransformer pipeline?
    if hasattr(model, "named_steps") and "feats" in model.named_steps:
        feats = model.named_steps["feats"]

        # Collect expected input columns (labels passed to ColumnTransformer)
        expected_cols = set()
        for name, trans, cols in feats.transformers_:
            if cols is None or cols == "drop":
                continue
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected_cols.update(cols)
            else:
                expected_cols.add(cols)

        # Build one-row DataFrame; fill everything with "" unless it's text_all
        row = {c: "" for c in expected_cols}
        row["text_all"] = text  # critical for our pipelines
        X_df = pd.DataFrame([row])
        return float(model.predict_proba(X_df)[:, 1])

    # Fallback: assume text list
    return float(model.predict_proba([text])[:, 1])

# ----------------------------- Sidebar: Data loading -----------------------------

st.sidebar.title("⚙️ Data")
csv_path = st.sidebar.text_input("CSV path", value="ticket_rag_data.csv")
uploaded = st.sidebar.file_uploader("…or upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error("CSV not found. Provide a valid path or upload a CSV.")
        st.stop()

st.success(f"Loaded {len(df):,} tickets")

# ----------------------------- Build RAG corpus -----------------------------

@st.cache_resource(show_spinner=False)
def fit_retriever(corpus_df: pd.DataFrame):
    text_clean = build_text_clean(corpus_df)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(text_clean)
    return tfidf, X

tfidf, X = fit_retriever(df)
st.caption(f"TF-IDF fitted on cleaned text (docs × terms = {X.shape[0]} × {X.shape[1]}).")

# Use expanded descriptions for display & summary
def retrieve(query: str, k: int = 5) -> pd.DataFrame:
    q_vec = tfidf.transform([clean_text_keep_case(query)])
    sims = cosine_similarity(q_vec, X).ravel()
    top_idx = np.argsort(-sims)[:k]

    hits = df.iloc[top_idx].copy()
    desc = hits.get("Ticket Description", pd.Series([""] * len(hits), index=hits.index)).astype(str)
    prod = hits.get("Product Purchased", None)
    desc_expanded = expand_product_placeholders_series(desc, prod)

    out = hits[["Ticket ID", "Ticket Type", "Ticket Subject"]].copy()
    out["Ticket Description (expanded)"] = desc_expanded.values
    out["similarity"] = sims[top_idx]
    return out

# ----------------------------- Tabs -----------------------------

tab_rag, tab_clf = st.tabs(["🔎 RAG Search + Summary", "🧭 Urgency Classifier"])

with tab_rag:
    st.header("Query tickets")
    q = st.text_input("Search query", value="refund not processed after cancellation")
    k = st.slider("Top K", 1, 10, 8)
    if st.button("Search", type="primary"):
        hits = retrieve(q, k=k)
        st.dataframe(hits, use_container_width=True)
        summary = extractive_summary(q, hits["Ticket Description (expanded)"].tolist(), max_sentences=3)
        st.subheader("Summary (extractive):")
        st.info(summary)

# ----------------------------- Classifier -----------------------------

with tab_clf:
    st.header("Train / use urgency classifier")

    # Try to load a pre-trained bundle if present
    models_path = st.sidebar.text_input("Pretrained models (optional)", value="urgency_models.joblib")
    bundle = None
    if os.path.exists(models_path):
        try:
            bundle = joblib_load(models_path)
            st.sidebar.success("Loaded urgency_models.joblib")
        except Exception as e:
            st.sidebar.warning(f"Could not load joblib: {e}")

    # Let user upload a training CSV for classifier if they want to train in-app
    clf_file = st.sidebar.file_uploader("Upload training CSV for classifier (optional)", type=["csv"], key="clf")
    clf_df = None
    if clf_file is not None:
        try:
            clf_df = pd.read_csv(clf_file)
        except Exception as e:
            st.warning(f"Could not read training CSV: {e}")

    # --- Helper: build labels from CSV
    def build_urgent_labels(frame: pd.DataFrame) -> pd.Series | None:
        """Prefer an 'urgent' column (0/1). Else derive from Ticket Priority if present."""
        if "urgent" in frame.columns:
            return pd.to_numeric(frame["urgent"], errors="coerce").fillna(0).astype(int)

        if "Ticket Priority" in frame.columns:
            pr = frame["Ticket Priority"].astype(str).str.lower()
            priority_map = {"critical": 1, "high": 1, "p1": 1, "urgent": 1, "medium": 0, "low": 0}
            return pr.map(priority_map).fillna(0).astype(int)

        return None

    # --- Helper: text column for classifier
    def build_text_all(frame: pd.DataFrame) -> pd.Series:
        desc = frame.get("Ticket Description", pd.Series([""] * len(frame), index=frame.index)).astype(str)
        subj = frame.get("Ticket Subject", pd.Series([""] * len(frame), index=frame.index)).astype(str)
        prod = frame.get("Product Purchased", None)
        desc_expanded = expand_product_placeholders_series(desc, prod)
        return (subj.fillna("") + " " + desc_expanded.fillna("")).apply(clean_text_keep_case)

    # --- Classifier builder (text only for robustness & speed)
    def make_clf_pipeline():
        tf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, stop_words="english", sublinear_tf=True)
        base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear")
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
        return Pipeline([("tfidf", tf), ("clf", clf)])

    # ---- Choose model source
    model = None
    threshold = 0.5
    validation_report = None
    confusion = None
    roc = None

    if bundle is not None:
        which = st.selectbox("Choose pre-trained model", ["strict (bundle)", "broad (bundle)"])
        if "strict" in which and "model_strict" in bundle:
            model = bundle["model_strict"]
            threshold = float(bundle.get("thr_strict", 0.5))
        elif "broad" in which and "model_broad" in bundle:
            model = bundle["model_broad"]
            threshold = float(bundle.get("thr_broad", 0.5))

        if model is not None:
            st.success(f"Classifier ready (threshold={threshold:.2f})")

    # Train from uploaded CSV if requested
    if st.checkbox("Train a classifier from uploaded CSV"):
        if clf_df is None:
            st.warning("Upload a training CSV in the sidebar.")
        else:
            y = build_urgent_labels(clf_df)
            if y is None:
                st.error("Training CSV must contain either an 'urgent' column or 'Ticket Priority'.")
            else:
                X_text = build_text_all(clf_df)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_text, y, test_size=0.2, random_state=42, stratify=y
                )
                pipe = make_clf_pipeline()
                with st.spinner("Training..."):
                    pipe.fit(X_train, y_train)
                p = pipe.predict_proba(X_test)[:,1]

                # Simple threshold selection by macro F2
                def fbeta(prec, rec, beta):
                    return (1+beta**2)*prec*rec / (beta**2*prec + rec + 1e-9)
                best_f, best_t = -1, 0.5
                best_rep, best_cm = None, None
                for t in np.linspace(0.2, 0.8, 13):
                    y_hat = (p >= t).astype(int)
                    cm = confusion_matrix(y_test, y_hat, labels=[0,1])
                    tn, fp, fn, tp = cm.ravel()
                    prec1 = tp / (tp + fp + 1e-9); rec1 = tp / (tp + fn + 1e-9)
                    prec0 = tn / (tn + fn + 1e-9); rec0 = tn / (tn + fp + 1e-9)
                    f1 = fbeta(prec1, rec1, 2.0); f0 = fbeta(prec0, rec0, 2.0)
                    fmacro = 0.5 * (f0 + f1)
                    if fmacro > best_f:
                        best_f, best_t = fmacro, t
                        best_rep = classification_report(y_test, y_hat, digits=3)
                        best_cm = cm

                model = pipe
                threshold = float(best_t)
                validation_report = best_rep
                confusion = best_cm
                roc = roc_auc_score(y_test, p)

                st.success(f"Classifier trained. Best threshold (macro F2): {threshold:.2f}; ROC-AUC {roc:.3f}")
                with st.expander("Validation report"):
                    st.text(validation_report)
                    st.text(f"Confusion matrix:\n{confusion}")

                if st.button("Save classifier as urgency_models.joblib"):
                    joblib_dump({"model_strict": model, "thr_strict": threshold}, "urgency_models.joblib")
                    st.success("Saved urgency_models.joblib")

    st.subheader("Predict urgency for a new ticket")
    new_subj = st.text_input("Ticket Subject", value="Refund request")
    new_desc = st.text_area(
        "Ticket Description",
        value="I'm having trouble with my refund after cancelling my order."
    )
    if st.button("Predict Urgency"):
        if model is None:
            st.warning("No classifier loaded or trained yet.")
        else:
            txt = clean_text_keep_case(new_subj + " " + new_desc)
            prob = predict_proba_safe(model, txt)   # <-- robust call
            pred = int(prob >= threshold)
            st.write(f"Urgent probability: **{prob:.3f}**  |  Threshold: **{threshold:.2f}**")
            st.success("Prediction: **URGENT**" if pred == 1 else "Prediction: **Non-urgent**")

# ----------------------------- Footer -----------------------------
st.caption("Built with scikit-learn + Streamlit • Notebook-aligned cleaning • No external APIs")

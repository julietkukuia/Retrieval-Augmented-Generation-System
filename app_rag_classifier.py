# app_rag_classifier.py
# Ticket RAG + Urgency Classifier (RAG + your EXACT ColumnTransformer pipeline)

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibrated import CalibratedClassifierCV  # sklearn 1.4/1.5
from joblib import load as joblib_load, dump as joblib_dump

st.set_page_config(page_title="Ticket RAG + Urgency Classifier", layout="wide")

# ==============================
# Utilities (cleaning & helpers)
# ==============================

def clean_text_keep_case(s: str) -> str:
    """Keep case & apostrophes (like notebook), trim, collapse spaces."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9'\s]", " ", s)
    return s

def expand_product_placeholders_series(desc: pd.Series, prod: pd.Series | None) -> pd.Series:
    """Replace {product_purchased} (any casing/underscore/space) with Product Purchased value."""
    if prod is None:
        return desc.fillna("").astype(str)
    rx = re.compile(r"\{product[_ ]purchased\}", flags=re.IGNORECASE)
    out = []
    for d, p in zip(desc.fillna("").astype(str), prod.fillna("").astype(str)):
        out.append(rx.sub(p, d))
    return pd.Series(out, index=desc.index)

def build_text_clean(df: pd.DataFrame) -> pd.Series:
    """Use df['text_clean'] if present; else Subject + expanded Description; keep case."""
    if "text_clean" in df.columns:
        return df["text_clean"].astype(str).fillna("").apply(clean_text_keep_case)

    desc = df.get("Ticket Description", pd.Series([""] * len(df), index=df.index)).astype(str)
    prod = df.get("Product Purchased", None)
    desc_expanded = expand_product_placeholders_series(desc, prod)
    subj = df.get("Ticket Subject", pd.Series([""] * len(df), index=df.index)).astype(str)
    return (subj.fillna("") + " " + desc_expanded.fillna("")).apply(clean_text_keep_case)

# ==============================
# RAG retriever (TF-IDF + cosine)
# ==============================

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

def extractive_summary(query: str, docs: list[str], max_sentences: int = 3) -> str:
    text = " ".join([str(d) for d in docs])
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "No summary available."

    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    S = vec.fit_transform(sentences)
    q = vec.transform([query])
    base = linear_kernel(q, S).ravel()

    chosen = []
    for _ in range(min(max_sentences, len(sentences))):
        best_i, best_score = None, -1e9
        for i in range(len(sentences)):
            if i in chosen:
                continue
            redundancy = 0.0 if not chosen else max(linear_kernel(S[i], S[chosen]).ravel())
            score = base[i] - 0.4 * redundancy
            if score > best_score:
                best_i, best_score = i, score
        chosen.append(best_i)
    return " ".join([sentences[i] for i in chosen])

# ==============================
# >>> YOUR EXACT CLASSIFIER PIPELINE <<<
# ==============================

def make_pipeline_with_priority(df_columns, use_numerics=True):
    transformers = [
        ('text',
         TfidfVectorizer(
             ngram_range=(1,2),
             min_df=2,
             max_df=0.95,
             token_pattern=r"(?u)\b\w[\w']*\b",
             sublinear_tf=True
         ),
         'text_all')
    ]

    cat_cols = [c for c in ['Ticket Channel','Ticket Type','Ticket Priority'] if c in df_columns]
    if cat_cols:
        transformers.append(('cats', OneHotEncoder(handle_unknown='ignore'), cat_cols))

    num_cols = []
    if use_numerics:
        for c in ['First Response Time (hrs)','Resolution Time (hrs)']:
            if c in df_columns:
                num_cols.append(c)
    if num_cols:
        transformers.append(('nums', StandardScaler(with_mean=False), num_cols))

    feats = ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)

    base = LogisticRegression(C=0.7, class_weight='balanced', max_iter=1000, solver='liblinear')
    clf  = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=5)

    return Pipeline([('feats', feats), ('clf', clf)])

def train_eval(df_labels, title, beta_for_recall=2.0):
    X = df_labels.drop(columns=['urgent'])
    y = df_labels['urgent'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = make_pipeline_with_priority(df_labels.columns, use_numerics=True)
    pipe.fit(X_train, y_train)

    p = pipe.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.2, 0.8, 13)
    best_f, best_t = -1, 0.5
    best_report, best_cm = None, None

    def fbeta(prec, rec, beta):
        return (1+beta**2)*prec*rec / (beta**2*prec + rec + 1e-9)

    for t in thresholds:
        y_hat = (p >= t).astype(int)
        cm = confusion_matrix(y_test, y_hat, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        prec1 = tp/(tp+fp+1e-9); rec1 = tp/(tp+fn+1e-9)
        prec0 = tn/(tn+fn+1e-9); rec0 = tn/(tn+fp+1e-9)
        f1 = fbeta(prec1, rec1, beta_for_recall)
        f0 = fbeta(prec0, rec0, beta_for_recall)
        f_macro = 0.5*(f0+f1)
        if f_macro > best_f:
            best_f, best_t = f_macro, t
            best_report = classification_report(y_test, y_hat, digits=3, zero_division=0)
            best_cm = cm

    st.write(f"**{title}**")
    st.write(f"ROC AUC: {roc_auc_score(y_test, p):.3f}")
    st.write(f"Best threshold F{beta_for_recall:.0f}: {best_t:.2f}")
    st.text(best_report)
    st.text(f"Confusion matrix:\n{best_cm}")

    return pipe, best_t

def build_text_all(frame: pd.DataFrame) -> pd.Series:
    """Create text_all for the ColumnTransformer pipeline (subject + expanded description)."""
    desc = frame.get("Ticket Description", pd.Series([""] * len(frame), index=frame.index)).astype(str)
    subj = frame.get("Ticket Subject", pd.Series([""] * len(frame), index=frame.index)).astype(str)
    prod = frame.get("Product Purchased", None)
    desc_expanded = expand_product_placeholders_series(desc, prod)
    return (subj.fillna("") + " " + desc_expanded.fillna("")).apply(clean_text_keep_case)

def ensure_classif_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the columns the pipeline expects exist."""
    f = frame.copy()
    if "text_all" not in f.columns:
        f["text_all"] = build_text_all(f)

    # Optional categoricals
    for c in ["Ticket Channel", "Ticket Type", "Ticket Priority"]:
        if c not in f.columns:
            f[c] = ""

    # Optional numerics
    for c in ["First Response Time (hrs)", "Resolution Time (hrs)"]:
        if c not in f.columns:
            f[c] = 0.0
        else:
            f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)

    return f

def predict_proba_bundle(model, row_df: pd.DataFrame) -> float:
    """Predict prob(urgent) from a ColumnTransformer pipeline using a 1-row DataFrame."""
    return float(model.predict_proba(row_df)[:, 1])

# ==============================
# UI: Tabs
# ==============================

tab_rag, tab_clf = st.tabs(["🔎 RAG Search + Summary", "🧭 Urgency Classifier"])

with tab_rag:
    st.header("Query tickets")
    q = st.text_input("Search query", value="refund not processed after cancellation")
    k = st.slider("Top K", 1, 10, 6)
    if st.button("Search", type="primary"):
        hits = retrieve(q, k=k)
        st.dataframe(hits, use_container_width=True)
        summary = extractive_summary(q, hits["Ticket Description (expanded)"].tolist(), max_sentences=3)
        st.subheader("Summary (extractive):")
        st.info(summary)

with tab_clf:
    st.header("Train / use urgency classifier")

    # Pretrained models
    models_path = st.sidebar.text_input("Pretrained models (optional)", value="urgency_models.joblib")
    bundle = None
    if os.path.exists(models_path):
        try:
            bundle = joblib_load(models_path)
            st.sidebar.success("Loaded urgency_models.joblib")
        except Exception as e:
            st.sidebar.warning(f"Could not load joblib: {e}")

    # Upload training CSV (optional)
    clf_file = st.sidebar.file_uploader("Upload training CSV for classifier (optional)", type=["csv"], key="clf")
    clf_df = None
    if clf_file is not None:
        try:
            clf_df = pd.read_csv(clf_file)
        except Exception as e:
            st.warning(f"Could not read training CSV: {e}")

    # Build labels
    def build_urgent_labels(frame: pd.DataFrame) -> pd.Series | None:
        # Prefer 'urgent' if present
        if "urgent" in frame.columns:
            return pd.to_numeric(frame["urgent"], errors="coerce").fillna(0).astype(int)

        # Else derive from Ticket Priority (strict mapping)
        if "Ticket Priority" in frame.columns:
            pr = frame["Ticket Priority"].astype(str).str.lower()
            priority_map = {"critical": 1, "high": 1, "p1": 1, "urgent": 1, "medium": 0, "low": 0}
            return pr.map(priority_map).fillna(0).astype(int)

        return None

    model = None
    threshold = 0.5
    if bundle is not None and "model_strict" in bundle:
        model = bundle["model_strict"]
        threshold = float(bundle.get("thr_strict", 0.5))
        st.success(f"Classifier ready (threshold={threshold:.2f})")

    # Train in-app with EXACT pipeline
    if st.checkbox("Train a classifier from uploaded CSV"):
        if clf_df is None:
            st.warning("Upload a training CSV in the sidebar.")
        else:
            y = build_urgent_labels(clf_df)
            if y is None:
                st.error("Training CSV must have either an 'urgent' column or 'Ticket Priority'.")
            else:
                df_train = ensure_classif_columns(clf_df)
                df_train = df_train.copy()
                df_train["urgent"] = y

                with st.spinner("Training (text + channel/type/priority + optional numerics)…"):
                    model, threshold = train_eval(
                        df_train,
                        "Calibrated LR on text + categories + numerics",
                        beta_for_recall=2.0
                    )

                # Save option
                if st.button("Save classifier as urgency_models.joblib"):
                    joblib_dump({"model_strict": model, "thr_strict": threshold}, "urgency_models.joblib")
                    st.success("Saved urgency_models.joblib")

    st.subheader("Predict urgency for a new ticket")

    colL, colR = st.columns(2)
    with colL:
        new_subj = st.text_input("Ticket Subject", value="Refund request")
        new_desc = st.text_area(
            "Ticket Description",
            value="I'm having trouble with my refund after cancelling my order."
        )
        new_channel = st.selectbox("Ticket Channel", ["", "email", "phone", "chat", "social media"])
        new_type = st.selectbox("Ticket Type", ["", "product inquiry", "billing inquiry", "refund request", "technical issue"])
        new_priority = st.selectbox("Ticket Priority", ["", "low", "medium", "high", "critical", "urgent"])
    with colR:
        frt = st.number_input("First Response Time (hrs)", min_value=0.0, value=0.0, step=0.5)
        rth = st.number_input("Resolution Time (hrs)", min_value=0.0, value=0.0, step=0.5)

    if st.button("Predict Urgency"):
        if model is None:
            st.warning("No classifier loaded or trained yet.")
        else:
            # Build 1-row DataFrame with the exact columns the pipeline expects
            row = pd.DataFrame([{
                "Ticket Subject": new_subj,
                "Ticket Description": new_desc,
                "Product Purchased": "",            # optional; used for expansion if present
                "Ticket Channel": new_channel,
                "Ticket Type": new_type,
                "Ticket Priority": new_priority,
                "First Response Time (hrs)": frt,
                "Resolution Time (hrs)": rth,
            }])

            row = ensure_classif_columns(row)  # creates text_all + fills missing cols
            prob = predict_proba_bundle(model, row)
            pred = int(prob >= threshold)

            st.write(f"Urgent probability: **{prob:.3f}**  |  Threshold: **{threshold:.2f}**")
            st.success("Prediction: **URGENT**" if pred == 1 else "Prediction: **Non-urgent**")

# -----------------------------
st.caption("Built with scikit-learn + Streamlit • RAG + Calibrated LR (ColumnTransformer) • No external APIs")

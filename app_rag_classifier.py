# app_rag_classifier.py
# Ticket RAG + Urgency Classifier (streamlit-safe + sklearn training + bundle upload)

import os
import io
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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import load as joblib_load, dump as joblib_dump

# Prefer calibration if available (works on most sklearn versions)
CALIB_AVAILABLE = True
try:
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    CalibratedClassifierCV = None
    CALIB_AVAILABLE = False

import sklearn

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
    return pd.Series([rx.sub(p or "", d or "") for d, p in zip(desc.fillna(""), prod.fillna(""))], index=desc.index)

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

# ---------- Prediction helper (works for text-only or ColumnTransformer pipelines) ----------

def _proba1_from_matrix(m: np.ndarray) -> float:
    return float(m[0, 1])

def predict_proba_safe(model, text: str) -> float:
    # text-only pipelines usually accept list[str]
    if hasattr(model, "predict_proba"):
        try:
            return _proba1_from_matrix(model.predict_proba([text]))
        except Exception:
            pass
    # ColumnTransformer pipelines expect a DataFrame with required columns
    if hasattr(model, "named_steps") and "feats" in model.named_steps:
        feats = model.named_steps["feats"]
        expected_cols = set()
        for name, trans, cols in feats.transformers_:
            if cols is None or cols == "drop":
                continue
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected_cols.update(cols)
            else:
                expected_cols.add(cols)
        row = {c: "" for c in expected_cols}
        row["text_all"] = text
        X_df = pd.DataFrame([row])
        return _proba1_from_matrix(model.predict_proba(X_df))
    # last resort
    return _proba1_from_matrix(model.predict_proba([text]))

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
        st.dataframe(hits, width="stretch")  # deprecation-safe
        summary = extractive_summary(q, hits["Ticket Description (expanded)"].tolist(), max_sentences=3)
        st.subheader("Summary (extractive):")
        st.info(summary)

# ----------------------------- Classifier -----------------------------

with tab_clf:
    st.header("Train / use urgency classifier")

    # ===== Load pre-trained bundle from disk (optional) =====
    models_path = st.sidebar.text_input("Pretrained bundle path", value="urgency_models.joblib")
    bundle = None
    if os.path.exists(models_path):
        try:
            bundle = joblib_load(models_path)
            v_saved = bundle.get("sklearn_version")
            if v_saved and v_saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                st.sidebar.warning(
                    f"Bundle trained on scikit-learn {v_saved}; server is {sklearn.__version__}. "
                    "If you see errors, retrain in-app or pin versions."
                )
            st.sidebar.success("Loaded bundle from disk")
        except Exception as e:
            st.sidebar.warning(f"Could not load joblib: {e}")

    # ===== Upload a pre-trained bundle in the UI =====
    uploaded_bundle = st.sidebar.file_uploader(
        "…or upload a trained model bundle (.joblib/.pkl)", type=["joblib", "pkl"], key="joblib_uploader"
    )
    if uploaded_bundle is not None:
        try:
            bytes_data = uploaded_bundle.read()
            maybe_bundle = joblib_load(io.BytesIO(bytes_data))
            if isinstance(maybe_bundle, dict):
                bundle = maybe_bundle
                v_saved = bundle.get("sklearn_version")
                if v_saved and v_saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                    st.sidebar.warning(
                        f"Uploaded bundle trained on scikit-learn {v_saved}; server is {sklearn.__version__}."
                    )
                st.sidebar.success("Uploaded model bundle loaded.")
            else:
                # Single pipeline: wrap with default threshold
                bundle = {"model_strict": maybe_bundle, "thr_strict": 0.5}
                st.sidebar.success("Uploaded single model, using default threshold 0.50.")
        except Exception as e:
            st.sidebar.error(f"Could not load uploaded bundle: {e}")

    # ===== Helpers for labels & text =====
    def build_urgent_labels(frame: pd.DataFrame) -> pd.Series | None:
        """Prefer explicit 'urgent'. Else derive from 'Ticket Priority'."""
        if "urgent" in frame.columns:
            return pd.to_numeric(frame["urgent"], errors="coerce").fillna(0).astype(int)
        if "Ticket Priority" in frame.columns:
            pr = frame["Ticket Priority"].astype(str).str.lower()
            priority_map = {"critical": 1, "high": 1, "p1": 1, "urgent": 1, "medium": 0, "low": 0}
            return pr.map(priority_map).fillna(0).astype(int)
        return None

    def build_text_all(frame: pd.DataFrame) -> pd.Series:
        desc = frame.get("Ticket Description", pd.Series([""] * len(frame), index=frame.index)).astype(str)
        subj = frame.get("Ticket Subject", pd.Series([""] * len(frame), index=frame.index)).astype(str)
        prod = frame.get("Product Purchased", None)
        desc_expanded = expand_product_placeholders_series(desc, prod)
        return (subj.fillna("") + " " + desc_expanded.fillna("")).apply(clean_text_keep_case)

    # ===== Pipelines (text-only) and (text + cats + nums with priority) =====
    def make_text_only_pipeline():
        tf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, stop_words="english", sublinear_tf=True)
        base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear")
        if CALIB_AVAILABLE:
            clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
            return Pipeline([("tfidf", tf), ("clf", clf)])
        return Pipeline([("tfidf", tf), ("clf", base)])

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
        if CALIB_AVAILABLE:
            clf  = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=5)
        else:
            clf = base
        return Pipeline([('feats', feats), ('clf', clf)])

    # ===== Choose model source =====
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
    # Allow manual threshold tweak for any loaded/trained model
    threshold = st.slider("Decision threshold (class 1 = urgent)", 0.05, 0.95, float(threshold), 0.01)

    # ===== Train in-app =====
    st.markdown("**Train a classifier (in-app)**")
    clf_file = st.sidebar.file_uploader("Upload training CSV (optional)", type=["csv"], key="clf")
    clf_df = None
    if clf_file is not None:
        try:
            clf_df = pd.read_csv(clf_file)
        except Exception as e:
            st.warning(f"Could not read training CSV: {e}")

    use_cats_nums = st.checkbox("Use categories + numerics (includes Ticket Priority as feature)", value=False)
    if use_cats_nums:
        st.info("If your **labels** were derived from Ticket Priority, including Ticket Priority as a **feature** may cause leakage. Prefer an explicit 'urgent' label column to avoid this.")

    if st.checkbox("Train now"):
        if clf_df is None:
            st.warning("Upload a training CSV in the sidebar.")
        else:
            # Build labels first
            y = build_urgent_labels(clf_df)
            if y is None:
                st.error("Training CSV must contain either an 'urgent' column or 'Ticket Priority'.")
            else:
                # Build features
                if use_cats_nums:
                    # prepare table with text_all + cats + nums
                    X = clf_df.copy()
                    X["text_all"] = build_text_all(clf_df)
                    # coerce numerics if present
                    for col in ['First Response Time (hrs)','Resolution Time (hrs)']:
                        if col in X.columns:
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    pipe = make_pipeline_with_priority(X.columns, use_numerics=True)
                else:
                    X_text = build_text_all(clf_df)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_text, y, test_size=0.2, random_state=42, stratify=y
                    )
                    pipe = make_text_only_pipeline()

                with st.spinner("Training..."):
                    pipe.fit(X_train, y_train)

                p = pipe.predict_proba(X_test)[:, 1]

                # threshold search by macro F2
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
                        best_rep = classification_report(y_test, y_hat, digits=3, zero_division=0)
                        best_cm = cm

                model = pipe
                threshold = float(best_t)  # update slider default visually via help text
                roc = roc_auc_score(y_test, p)

                st.success(f"Trained. Best threshold (macro F2): {threshold:.2f} | ROC-AUC: {roc:.3f}")
                with st.expander("Validation report"):
                    st.text(best_rep)
                    st.text(f"Confusion matrix:\n{best_cm}")
                if not CALIB_AVAILABLE:
                    st.info("Note: CalibratedClassifierCV not available; using plain LogisticRegression probabilities.")

                # Save bundle button
                if st.button("Save as urgency_models.joblib"):
                    jb = {
                        "model_strict": model,
                        "thr_strict": float(threshold),
                        "sklearn_version": sklearn.__version__,
                    }
                    joblib_dump(jb, "urgency_models.joblib")
                    st.success("Saved urgency_models.joblib")

    # ===== Predict =====
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
            prob = predict_proba_safe(model, txt)
            pred = int(prob >= threshold)
            st.write(f"Urgent probability: **{prob:.3f}**  |  Threshold: **{threshold:.2f}**")
            st.success("Prediction: **URGENT**" if pred == 1 else "Prediction: **Non-urgent**")

# ----------------------------- Footer -----------------------------
st.caption("Built with scikit-learn + Streamlit • Notebook-aligned cleaning • ColumnTransformer option • Bundle upload")

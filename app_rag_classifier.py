# app_rag_classifier.py — "fastest easy way out" edition
import os, re, numpy as np, pandas as pd, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
try:
    # correct path on modern scikit-learn (e.g. 1.7.x)
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    # very old versions fallback (won't be used on Streamlit Cloud)
    from sklearn.calibrated import CalibratedClassifierCV
from joblib import load as joblib_load, dump as joblib_dump

st.set_page_config(page_title="Ticket RAG + Urgency Classifier", layout="wide")

# ---------- tiny helpers ----------
def clean_text_keep_case(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9'\s]", " ", s)
    return s

def expand_product_placeholders_series(desc: pd.Series, prod: pd.Series | None) -> pd.Series:
    if prod is None:
        return desc.fillna("").astype(str)
    rx = re.compile(r"\{product[_ ]purchased\}", flags=re.IGNORECASE)
    return pd.Series([rx.sub(p or "", d or "") for d, p in zip(desc, prod)], index=desc.index)

def build_text_clean(df: pd.DataFrame) -> pd.Series:
    # honor precomputed clean column if present
    if "text_clean" in df.columns:
        return df["text_clean"].astype(str).fillna("").apply(clean_text_keep_case)
    desc = df.get("Ticket Description", pd.Series([""] * len(df), index=df.index)).astype(str)
    subj = df.get("Ticket Subject", pd.Series([""] * len(df), index=df.index)).astype(str)
    prod = df.get("Product Purchased", None)
    return (subj.fillna("") + " " + expand_product_placeholders_series(desc, prod).fillna("")).apply(clean_text_keep_case)

def extractive_summary(query: str, docs: list[str], max_sentences: int = 3) -> str:
    text = " ".join([str(d) for d in docs])
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
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

def predict_proba_safe(model, text: str) -> float:
    # Text-only pipeline?
    if hasattr(model, "named_steps") and "tfidf" in model.named_steps:
        return float(model.predict_proba([text]).ravel()[1])
    # ColumnTransformer pipeline?
    if hasattr(model, "named_steps") and "feats" in model.named_steps:
        feats = model.named_steps["feats"]
        expected_cols = set()
        for _, _, cols in feats.transformers_:
            if cols is None or cols == "drop":
                continue
            if isinstance(cols, (list, tuple, np.ndarray)):
                expected_cols.update(cols)
            else:
                expected_cols.add(cols)
        row = {c: "" for c in expected_cols}
        row["text_all"] = text
        return float(model.predict_proba(pd.DataFrame([row])).ravel()[1])
    # Fallback
    return float(model.predict_proba([text]).ravel()[1])

# ---------- data load ----------
st.sidebar.title("⚙️ Data")
csv_path = st.sidebar.text_input("CSV path", value="ticket_rag_data.csv")
uploaded = st.sidebar.file_uploader("…or upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    st.error("CSV not found. Provide a valid path or upload a CSV.")
    st.stop()
st.success(f"Loaded {len(df):,} tickets")

# ---------- retriever ----------
@st.cache_resource(show_spinner=False)
def fit_retriever(corpus_df: pd.DataFrame):
    text_clean = build_text_clean(corpus_df)
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=2, max_df=0.9, sublinear_tf=True)
    X = tfidf.fit_transform(text_clean)
    return tfidf, X
tfidf, X = fit_retriever(df)
st.caption(f"TF-IDF fitted (docs × terms = {X.shape[0]} × {X.shape[1]}).")

def retrieve(query: str, k: int = 5) -> pd.DataFrame:
    sims = cosine_similarity(tfidf.transform([clean_text_keep_case(query)]), X).ravel()
    idx = np.argsort(-sims)[:k]
    hits = df.iloc[idx].copy()
    desc = hits.get("Ticket Description", pd.Series([""] * len(hits), index=hits.index)).astype(str)
    prod = hits.get("Product Purchased", None)
    hits_out = hits[["Ticket ID","Ticket Type","Ticket Subject"]].copy()
    hits_out["Ticket Description (expanded)"] = expand_product_placeholders_series(desc, prod).values
    hits_out["similarity"] = sims[idx]
    return hits_out

# ---------- UI ----------
tab_rag, tab_clf = st.tabs(["🔎 RAG Search + Summary", "🧭 Urgency Classifier"])

with tab_rag:
    st.header("Query tickets")
    q = st.text_input("Search query", value="refund not processed after cancellation")
    k = st.slider("Top K", 1, 10, 6)
    if st.button("Search", type="primary"):
        hits = retrieve(q, k=k)
        st.dataframe(hits, width="stretch")
        summary = extractive_summary(q, hits["Ticket Description (expanded)"].tolist(), max_sentences=3)
        st.subheader("Summary (extractive):")
        st.info(summary)

with tab_clf:
    st.header("Train / use urgency classifier")

    # Optional: try to load a pre-trained bundle, but ignore incompatibilities
    models_path = st.sidebar.text_input("Pretrained models (optional)", value="urgency_models.joblib")
    bundle, model, threshold = None, None, 0.5
    if os.path.exists(models_path):
        try:
            bundle = joblib_load(models_path)
            which = st.selectbox("Choose pre-trained model", ["strict (bundle)", "broad (bundle)"])
            if "strict" in which and "model_strict" in bundle:
                model = bundle["model_strict"]; threshold = float(bundle.get("thr_strict", 0.5))
            elif "broad" in which and "model_broad" in bundle:
                model = bundle["model_broad"];  threshold = float(bundle.get("thr_broad", 0.5))
            st.sidebar.success("Loaded urgency_models.joblib")
        except Exception as e:
            st.sidebar.warning(f"Ignoring incompatible model file: {e}")

    clf_file = st.sidebar.file_uploader("Upload training CSV for classifier (optional)", type=["csv"], key="clf")

    def build_urgent_labels(frame: pd.DataFrame) -> pd.Series | None:
        if "urgent" in frame.columns:
            return pd.to_numeric(frame["urgent"], errors="coerce").fillna(0).astype(int)
        if "Ticket Priority" in frame.columns:
            pr = frame["Ticket Priority"].astype(str).str.lower()
            mapping = {"critical":1,"high":1,"urgent":1,"p1":1,"medium":0,"low":0}
            return pr.map(mapping).fillna(0).astype(int)
        return None

    def build_text_all(frame: pd.DataFrame) -> pd.Series:
        desc = frame.get("Ticket Description", pd.Series([""] * len(frame), index=frame.index)).astype(str)
        subj = frame.get("Ticket Subject", pd.Series([""] * len(frame), index=frame.index)).astype(str)
        prod = frame.get("Product Purchased", None)
        return (subj.fillna("") + " " + expand_product_placeholders_series(desc, prod).fillna("")).apply(clean_text_keep_case)

    def make_clf_pipeline():
        tf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, stop_words="english", sublinear_tf=True)
        base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear")
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
        return Pipeline([("tfidf", tf), ("clf", clf)])

    if st.checkbox("Train a classifier from uploaded CSV"):
        if clf_file is None:
            st.warning("Upload a training CSV in the sidebar.")
        else:
            try:
                clf_df = pd.read_csv(clf_file)
            except Exception as e:
                st.error(f"Could not read training CSV: {e}")
                st.stop()

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
                def fbeta(prec, rec, beta): return (1+beta**2)*prec*rec / (beta**2*prec + rec + 1e-9)
                best_f, best_t, best_rep, best_cm = -1, 0.5, None, None
                for t in np.linspace(0.2, 0.8, 13):
                    y_hat = (p >= t).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0,1]).ravel()
                    prec1 = tp/(tp+fp+1e-9); rec1 = tp/(tp+fn+1e-9)
                    prec0 = tn/(tn+fn+1e-9); rec0 = tn/(tn+fp+1e-9)
                    f1 = fbeta(prec1, rec1, 2.0); f0 = fbeta(prec0, rec0, 2.0)
                    fmacro = 0.5*(f0+f1)
                    if fmacro > best_f:
                        best_f, best_t = fmacro, t
                        best_rep = classification_report(y_test, y_hat, digits=3, zero_division=0)
                        best_cm = np.array([[tn, fp],[fn, tp]])
                model, threshold = pipe, float(best_t)
                roc = roc_auc_score(y_test, p)
                st.success(f"Classifier trained. Best threshold (macro F2): {threshold:.2f}; ROC-AUC {roc:.3f}")
                with st.expander("Validation report"):
                    st.text(best_rep); st.text(f"Confusion matrix:\n{best_cm}")
                if st.button("Save classifier as urgency_models.joblib"):
                    joblib_dump({"model_strict": model, "thr_strict": threshold}, "urgency_models.joblib")
                    st.success("Saved urgency_models.joblib")

    st.subheader("Predict urgency for a new ticket")
    new_subj = st.text_input("Ticket Subject", value="Refund request")
    new_desc = st.text_area("Ticket Description", value="I'm having trouble with my refund after cancelling my order.")
    if st.button("Predict Urgency"):
        if model is None:
            st.warning("No classifier loaded or trained yet.")
        else:
            txt = clean_text_keep_case(new_subj + " " + new_desc)
            prob = predict_proba_safe(model, txt)
            pred = int(prob >= threshold)
            st.write(f"Urgent probability: **{prob:.3f}**  |  Threshold: **{threshold:.2f}**")
            st.success("Prediction: **URGENT**" if pred == 1 else "Prediction: **Non-urgent**")

st.caption("Built with scikit-learn + Streamlit • Safe imports • Trains in-app if needed")

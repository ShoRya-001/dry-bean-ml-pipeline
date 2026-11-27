"""
Interactive Dry Bean Variety Classifier Demo (Streamlit + CLI Verification)
=========================================================================
Powered by the serialized leak-free scikit-learn Stacking Ensemble pipeline
trained on the UCI Dry Bean Dataset (Koklu & Ozkan, 2020; Dataset ID: 602).

Run Web UI:
    pip install -r requirements.txt
    streamlit run app.py

Run CLI Sanity Check:
    python3 app.py --verify
"""

import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = os.path.join("artifacts", "best_dry_bean_pipeline.joblib")
METRICS_PATH = os.path.join("artifacts", "metrics_summary.json")

RAW_COLUMN_ALIASES = {
    "AspectRation": "AspectRatio",
    "roundness": "Roundness",
}

FEATURE_DESCRIPTIONS = {
    "Area": "Pixel area inside grain boundary (A)",
    "Perimeter": "Grain circumference / border length (P)",
    "MajorAxisLength": "Longest axis across grain (L)",
    "MinorAxisLength": "Longest perpendicular axis (l)",
    "AspectRatio": "Elongation ratio L / l",
    "Eccentricity": "Ellipse eccentricity matching second moments",
    "ConvexArea": "Pixel area of smallest convex polygon (C)",
    "EquivDiameter": "Diameter of equal-area circle: sqrt(4A/pi)",
    "Extent": "Ratio of grain area to bounding-box area",
    "Solidity": "Convexity ratio A / C",
    "Roundness": "Circularity metric: (4*pi*A) / P^2",
    "Compactness": "Roundness measure: EquivDiameter / MajorAxisLength",
    "ShapeFactor1": "MajorAxisLength / Area",
    "ShapeFactor2": "MinorAxisLength / Area",
    "ShapeFactor3": "4*A / (pi * MajorAxisLength^2)",
    "ShapeFactor4": "4*A / (pi * MajorAxisLength * MinorAxisLength)",
}


def load_artifacts(artifact_path: str = ARTIFACT_PATH):
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Missing serialized model at '{artifact_path}'. "
            "Run Dry_Bean_End_to_End_ML_Pipeline.ipynb first to export artifacts."
        )
    bundle = joblib.load(artifact_path)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return bundle, metrics


def normalize_input_df(df_in: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df_clean = df_in.rename(columns=RAW_COLUMN_ALIASES).copy()
    missing = [c for c in feature_names if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Missing required morphological feature columns: {missing}")
    return df_clean[feature_names].astype(float)


def predict_beans(bundle: dict, df_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pipe = bundle["pipeline"]
    class_names = bundle["class_names"]
    X_aligned = normalize_input_df(df_features, bundle["feature_names"])
    pred_ids = pipe.predict(X_aligned)
    pred_labels = np.array([class_names[int(i)] for i in pred_ids])
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_aligned)
    else:
        probs = np.eye(len(class_names))[pred_ids]
    return pred_labels, probs


def run_cli_verification() -> int:
    bundle, metrics = load_artifacts()
    presets = bundle["sample_presets"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]

    print(f"Loaded Model : {bundle['model_name']}")
    print(f"Test Accuracy: {bundle['test_accuracy']:.4f} | Test Macro-F1: {bundle['test_macro_f1']:.4f}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print("-" * 72)

    preset_rows = []
    for true_cls in class_names:
        row = presets[true_cls]
        preset_rows.append(row)
    df_presets = pd.DataFrame(preset_rows)[feature_names]
    pred_labels, probs = predict_beans(bundle, df_presets)

    correct = 0
    for idx, true_cls in enumerate(class_names):
        p_cls = pred_labels[idx]
        conf = float(probs[idx].max()) * 100.0
        prob_sum = float(probs[idx].sum())
        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities must sum to 1, got {prob_sum}"
        match_flag = "OK" if p_cls == true_cls else "DIFF"
        if p_cls == true_cls:
            correct += 1
        print(f"True: {true_cls:10s} | Predicted: {p_cls:10s} | Conf: {conf:6.2f}% | [{match_flag}]")

    print("-" * 72)
    print(f"CLI Verification Passed | Preset Match: {correct}/{len(class_names)} varieties | Metrics loaded: {bool(metrics)}")
    return 0


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Dry Bean Variety Classifier (UCI 602)",
        page_icon="🌱",
        layout="wide",
    )

    bundle, metrics = load_artifacts()
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]
    bounds = bundle["feature_bounds"]
    presets = bundle["sample_presets"]

    st.title("Dry Bean Morphological Classifier (End-to-End ML Pipeline)")
    st.caption(
        "UCI Machine Learning Repository (Dataset ID: 602) — Koklu & Ozkan (2020) | "
        f"Deployed Model: **{bundle['model_name']}**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployed Model", bundle["model_name"])
    c2.metric("Hold-Out Test Accuracy", f"{bundle['test_accuracy']*100:.2f}%")
    c3.metric("Hold-Out Macro-F1", f"{bundle['test_macro_f1']*100:.2f}%")
    c4.metric("Classes / Features", f"{len(class_names)} Varieties / {len(feature_names)} Features")

    tab_single, tab_batch, tab_card = st.tabs([
        "Single Grain Predictor", "Batch CSV Inference", "Dataset & Model Card"
    ])

    with tab_single:
        st.subheader("Interactive Feature Input")
        preset_choice = st.selectbox(
            "Load a real unseen Hold-Out Test Sample Preset (or tweak values below):",
            options=class_names,
            index=0,
        )
        active_preset = presets[preset_choice]

        col_dim, col_shape = st.columns(2)
        user_values = {}

        dim_cols = feature_names[:8]
        shape_cols = feature_names[8:]

        with col_dim:
            st.markdown("#### Dimensional Measurements (8)")
            for col in dim_cols:
                b = bounds[col]
                step = max((b["max"] - b["min"]) / 200.0, 1e-4)
                user_values[col] = st.number_input(
                    f"{col}",
                    min_value=float(b["min"] * 0.5),
                    max_value=float(b["max"] * 1.5),
                    value=float(active_preset[col]),
                    step=float(round(step, 5)),
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS[col],
                    key=f"inp_{col}_{preset_choice}",
                )

        with col_shape:
            st.markdown("#### Shape Ratios & Factors (8)")
            for col in shape_cols:
                b = bounds[col]
                step = max((b["max"] - b["min"]) / 200.0, 1e-5)
                user_values[col] = st.number_input(
                    f"{col}",
                    min_value=float(max(0.0, b["min"] * 0.5)),
                    max_value=float(b["max"] * 1.5),
                    value=float(active_preset[col]),
                    step=float(round(step, 6)),
                    format="%.5f",
                    help=FEATURE_DESCRIPTIONS[col],
                    key=f"inp_{col}_{preset_choice}",
                )

        input_df = pd.DataFrame([user_values])[feature_names]
        pred_labels, probs = predict_beans(bundle, input_df)
        pred_cls = pred_labels[0]
        prob_vec = probs[0]
        sorted_idx = np.argsort(prob_vec)[::-1]
        top1_cls, top1_p = class_names[sorted_idx[0]], float(prob_vec[sorted_idx[0]])
        top2_cls, top2_p = class_names[sorted_idx[1]], float(prob_vec[sorted_idx[1]])

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.success(f"### Predicted Variety: **{pred_cls}**")
            st.metric("Prediction Confidence", f"{top1_p * 100:.2f}%", delta=f"+{(top1_p - top2_p)*100:.2f}% over {top2_cls}")
            st.write(f"**Loaded Preset Ground Truth:** `{preset_choice}`")
        with res_col2:
            prob_df = pd.DataFrame({
                "Variety": class_names,
                "Probability": prob_vec,
            }).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Variety"))

    with tab_batch:
        st.subheader("Batch CSV Grain Classification")
        template_df = pd.DataFrame([presets[c] for c in class_names])[feature_names]
        st.download_button(
            "Download Sample 7-Bean CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="dry_bean_sample_template.csv",
            mime="text/csv",
        )
        uploaded = st.file_uploader("Upload CSV with 16 Dry Bean Morphological Features", type=["csv"])
        if uploaded is not None:
            raw_batch = pd.read_csv(uploaded)
            try:
                preds, probs = predict_beans(bundle, raw_batch)
                out_df = raw_batch.copy()
                out_df["Predicted_Class"] = preds
                out_df["Confidence"] = np.round(probs.max(axis=1), 4)
                st.success(f"Scored {len(out_df):,} grain records successfully!")
                st.dataframe(out_df.head(50), use_container_width=True)
                st.download_button(
                    "Download Scored Predictions CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="dry_bean_scored_predictions.csv",
                    mime="text/csv",
                )
            except Exception as err:
                st.error(f"Input validation error: {err}")

    with tab_card:
        st.subheader("Model & Dataset Provenance")
        st.markdown("""
        * **Official Source:** UCI Machine Learning Repository, Dataset ID **602**  
          `https://archive.ics.uci.edu/dataset/602/dry+bean+dataset`
        * **Citation:** Koklu, M., & Ozkan, I. A. (2020). *"Multiclass classification of dry beans using computer vision and machine learning techniques."* Computers and Electronics in Agriculture, 174, 105507.
        * **Leak-Free Pipeline:** `SimpleImputer(median)` + `StandardScaler()` encapsulated inside `sklearn.pipeline.Pipeline` paired with a `StackingClassifier` combining tuned `SVC (RBF)`, `HistGradientBoostingClassifier`, `MLPClassifier`, `RandomForestClassifier`, and `LogisticRegression` via a `LogisticRegression` meta-learner.
        """)
        if "master_test_table" in metrics:
            st.markdown("#### Hold-Out Test Benchmark Summary")
            st.dataframe(pd.DataFrame(metrics["master_test_table"]), use_container_width=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--verify", action="store_true")
    args, _ = parser.parse_known_args()
    if args.verify:
        sys.exit(run_cli_verification())
    run_streamlit_app()

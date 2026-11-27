# End-to-End Leak-Free Machine Learning: Dry Bean Multiclass Variety Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo_App-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/UCI_ML_Repository-ID_602-teal.svg)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

A complete, reproducible, zero-leakage tabular Machine Learning project built on the **UCI Dry Bean Dataset** (`13,611` raw samples $\rightarrow$ `13,543` unique grains, `16` continuous computer-vision morphological features, `7` bean varieties). Designed to run end-to-end on **Google Colab Free Tier CPU** (~11 minutes total runtime), backed by a **14-page Viva & Mathematical Mechanics Guide PDF** and an interactive **Streamlit Web Demo**.

> **Live Demo:** `[Replace with your Streamlit URL, e.g., https://your-app.streamlit.app]`

---

## Project Deliverables & Repository Structure

```text
.
├── app.py                                   # Interactive Streamlit demo & CLI --verify harness
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
├── Dry_Bean_End_to_End_ML_Pipeline.ipynb    # Executed Colab-ready notebook (41 cells, 17 plots)
├── Dry_Bean_ML_Viva_and_Code_Guide.pdf      # 14-page Viva Q&A (42 Q&As), Requirement Map & Math Guide
├── artifacts/
│   ├── best_dry_bean_pipeline.joblib        # Serialized CV-winning Stacking_Ensemble pipeline (17 MB)
│   └── metrics_summary.json                 # Complete JSON ledger of CV, hold-out, XAI & dataset stats
└── data/
    └── Dry_Bean_Dataset.xlsx                # Official raw UCI Dry Bean Excel dataset (13,611 x 17)
```

---

## Dataset Provenance & Citation

- **Source:** [UCI Machine Learning Repository — Dry Bean Dataset (ID: 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Official Citation:**
  > Koklu, M., & Ozkan, I. A. (2020). *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.* **Computers and Electronics in Agriculture**, 174, 105507. DOI: `10.1016/j.compag.2020.105507`
- **Data Hygiene & Split (`RANDOM_STATE = 42`):**
  - Raw shape: `13,611` rows $\times$ `17` columns (`0` missing values).
  - Pre-split deduplication: Removed `68` exact duplicate `HOROZ` rows before splitting $\rightarrow$ **`13,543` clean unique grains**.
  - Column typos standardized: `AspectRation` $\rightarrow$ `AspectRatio`, `roundness` $\rightarrow$ `Roundness`.
  - Stratified 80/20 split: **`10,834` Train** / **`2,709` Locked Hold-Out Test** (`6.79 : 1` imbalance across `DERMASON` 3,546 down to `BOMBAY` 522).

---

## Key Methodological Highlights (Zero Data Leakage)

1. **Strict `Pipeline` & `ColumnTransformer` Encapsulation:** Median imputation, `StandardScaler` (and `MinMaxScaler` branch for `MultinomialNB`), `PCA`, `SMOTE`, and wrapper feature selectors are fit strictly inside 5-Fold Stratified CV training folds—the 2,709-row test set is evaluated once after model selection.
2. **Unsupervised Structure Audit (Train $X$):**
   - **K-Means++** ($k=2..12$): Silhouette peak at $k=3$ (`0.4066`, ARI `0.2976`); at $k=7$, ARI reaches **`0.6668`** (NMI `0.7118`).
   - **DBSCAN** (`min_samples=16`): $k$-distance knee $\varepsilon=1.2$ chains touching classes (`2` clusters, `5.56%` noise, ARI `0.0387`), while $\varepsilon=0.7$ isolates `4` dense cores (`37.28%` noise, ARI `0.2413`).
3. **Full Scikit-Learn Benchmark (`15` Models / `9` Families):** Evaluates Linear (`LogisticRegression`, `RidgeClassifier`), Discriminant (`LDA`), Distance (`KNN`), Probabilistic (`GaussianNB`, `MultinomialNB`), Kernel/Margin (`LinearSVC`, `SVC RBF`), Tree (`DecisionTree`), Bagging (`RandomForest`, `ExtraTrees`), Boosting (`AdaBoost`, stochastic `GradientBoosting`, `HistGradientBoosting`), and Neural (`MLPClassifier`).
4. **Mandatory Forward & Backward `SequentialFeatureSelector` + `PCA`:**
   - Outer fold-safe Forward & Backward SFS reduce inputs by **50%** (`16` $\rightarrow$ `8` features), with **Backward SFS reaching `94.22%` CV Macro-F1** on `SVC` (outperforming all 16 untuned features).
   - `PCA` captures `95.04%` variance in `4` PCs (`89.55%` CV F1) and `99.61%` in `7` PCs (`94.11%` CV F1).
5. **Imbalance, Tuning & Multi-Family Ensembling:** Compares Baseline vs. `class_weight='balanced'` vs. fold-safe `SMOTE` (`imblearn.pipeline.Pipeline`), tunes `SVC`, `HistGradientBoosting`, and `MLPClassifier` via `RandomizedSearchCV`, and fuses 5 complementary families via **Hard Voting**, **Soft Voting**, and **`StackingClassifier`** (`LogisticRegression` meta-learner).
6. **Diagnostics & Explainability:** OvR ROC & Precision-Recall curves, Learning Curves, Hold-Out **Permutation Importance**, and **TreeSHAP**.

---

## Summary of Experimental Results

### 5-Fold Stratified CV Highlights (`n = 10,834` Train)
| Model / Regime | CV Accuracy | CV Macro-F1 | OvR ROC-AUC |
|---|---:|---:|---:|
| **`Stacking_Ensemble` (CV Winner)** | **93.21%** | **94.37%** | 0.9949 |
| `Voting_Hard` | 93.17% | 94.31% | — |
| `Voting_Soft` | 93.14% | 94.30% | 0.9959 |
| `Backward_SFS_8` (`SVC RBF`, 8 feats) | 93.07% | 94.22% | — |
| `Forward_SFS_8` (`SVC RBF`, 8 feats) | 92.98% | 94.16% | — |
| `SVC (RBF)` (All-16 Baseline) | 93.00% | 94.16% | 0.9955 |
| `HistGradientBoosting` (All-16 Baseline) | 92.83% | 94.05% | 0.9952 |
| `AdaBoost` (Depth-1 Stumps) | 79.74% | 69.45% | 0.9672 |
| `MultinomialNB` (`MinMaxScaler`) | 51.80% | 43.34% | 0.9590 |

### Hold-Out Test Evaluation (`n = 2,709` Locked Test)
| Candidate Pipeline | Features | Test Accuracy | Test Macro-F1 | OvR ROC-AUC |
|---|---:|---:|---:|---:|
| `SVC_RBF_Tuned` | 16 | 92.43% | 93.61% | 0.9947 |
| `Backward_SFS_8 (SVC)` | 8 | 92.47% | 93.58% | 0.9945 |
| `Forward_SFS_8 (SVC)` | 8 | 92.47% | 93.56% | 0.9945 |
| **`Stacking_Ensemble` [CV Winner]** | **16** | **92.40%** | **93.50%** | **0.9938** |
| `MLP_Tuned` | 16 | 92.36% | 93.35% | 0.9951 |
| `HistGB_Tuned` | 16 | 91.62% | 92.88% | 0.9950 |
| `PCA_95%_4PC (SVC)` | 4 | 89.63% | 89.70% | 0.9906 |

- **Per-Class Test F1 (`Stacking_Ensemble`):** `BOMBAY` **100.00%**, `HOROZ` **95.52%**, `SEKER` **94.99%**, `CALI` **93.13%**, `DERMASON` **91.93%**, `BARBUNYA` **91.60%**, `SIRA` **87.30%**.
- **Top XAI Drivers:** Permutation Importance (`Stacking`): `ShapeFactor4` (`0.0834`), `Roundness` (`0.0749`), `ShapeFactor1` (`0.0670`). Mean `|SHAP|` (`HistGB_Tuned`): `Compactness` (`0.4449`), `Area` (`0.4239`), `ShapeFactor1` (`0.3164`).

---

## Quickstart: Run Locally

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/dry-bean-ml-pipeline.git
cd dry-bean-ml-pipeline
pip install -r requirements.txt
```
*(Note: `artifacts/best_dry_bean_pipeline.joblib` was serialized with `scikit-learn==1.6.1`. If deploying to Streamlit Cloud, pinning `scikit-learn==1.6.1` in `requirements.txt` prevents pickle version mismatches.)*

### 2. Verify Inference End-to-End (CLI)
```bash
python3 app.py --verify
```
Expected output: `7/7` hold-out variety presets matched (`95.86%–99.60%` confidence).

### 3. Launch the Interactive Streamlit Web Demo
```bash
streamlit run app.py
```
The app provides three tabs:
- **Single Grain Predictor:** Interactive controls for all 16 features, 7 real hold-out variety presets, and live class probability bars.
- **Batch CSV Inference:** Upload bulk CSV feature files (auto-normalizes raw UCI column typos like `AspectRation` and `roundness`) and download predictions.
- **Dataset & Model Card:** Interactive inspection of dataset summary, 5-fold CV results, hold-out metrics, and feature definitions.

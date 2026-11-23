# Centrality-Driven Graph Models for Cancer Protein Prediction in PPI Networks

An end-to-end, reproducible bioinformatics and graph machine learning research platform for prioritizing potentially cancer-associated proteins in human Protein-Protein Interaction (PPI) networks.

Built as an independent, modernized, leak-free successor inspired by the research task in [`sevmardi/Protein-Protein-Networks`](https://github.com/sevmardi/Protein-Protein-Networks) (whose original source code remains unmodified in `../reference/Protein-Protein-Networks/`).

---

## Key Highlights & Verified Experimental Results

### 1. Real Biological Dataset (STRING v12.5 + COSMIC Cancer Gene Census v104)
- **PPI Network:** STRING v12.5 *Homo sapiens* (`combined_score >= 700` high-confidence threshold)
  - **17,525** interacting human proteins | **282,864** deduplicated undirected edges (`282,864` reverse duplicates removed; `0` self-loops; `2,174` isolated STRING info proteins excluded)
  - **82** connected components (Giant Connected Component = **98.92%**, `17,336` proteins)
  - Average degree = **32.28** | Cancer proteins avg degree = **73.70 (median 48.0)** vs Non-cancer avg degree = **30.39 (median 14.0)**
- **Cancer Labels:** COSMIC Cancer Gene Census v104 GRCh38 (Tier 1 & Tier 2) -> **767** mapped cancer driver proteins (**4.38%** minority class) vs **16,758** non-cancer proteins.
- **Anti-Leakage Controls:**
  - All cancer/tumor terms (`cancer`, `tumor`, `oncogene`, `carcinoma`, `leukemia`, etc.) are scrubbed from STRING functional annotation text before TF-IDF/SVD.
  - Strict stratified Train (`70%`, `n=12,267`) / Validation (`15%`, `n=2,629`) / Test (`15%`, `n=2,629`) splits.
  - All scalers, TF-IDF/SVD vectorizers, GNN training losses, threshold tuning, probability calibrators, and composite centrality weights are fit **strictly without test labels**.
- **Original Schema Compatibility:** Separate configuration (`config/dataset_original_schema.yaml`) supports `HumanPPI.txt`, `Functions.txt`, `Cancers.txt`, `Test1.txt`, and `Test2.txt` if an authorized copy of the private original dataset is supplied.

### 2. Model Comparison (Held-Out Test Split, Random PR-AUC Baseline = `0.0438`)

| Model | Features | Val PR-AUC | **Test PR-AUC** | Test ROC-AUC | Test F1 | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GraphSAGE (PyTorch)** *(Selected)* | Centrality + Functional + Topology | **0.3150** | **0.4102** | **0.8581** | **0.4127** | 0.3796 | 0.4522 |
| **GCN (PyTorch)** | Centrality + Functional + Topology | 0.2954 | **0.4060** | 0.8499 | 0.3871 | 0.3293 | 0.4696 |
| **Centrality-Driven Fusion GNN** | Centrality + Functional + Dual GraphSAGE/GCN | 0.2948 | **0.3923** | 0.8527 | 0.3871 | **0.4118** | 0.3652 |
| **Random Forest (Combined Graph Fusion)** | Centrality + Functional + Topology + GNN Emb | 0.2598 | 0.3487 | 0.8513 | 0.3653 | 0.3846 | 0.3478 |
| **GAT (PyTorch)** | Centrality + Functional + Topology | 0.2470 | 0.3289 | 0.8486 | 0.3423 | 0.3551 | 0.3304 |
| **Centrality-Driven Ensemble (XGB+GNN)** | Centrality + Functional + Topology + GNN Fusion | 0.2860 | 0.3193 | 0.8508 | 0.3775 | 0.3507 | 0.4087 |
| Reference WNC Heuristic (Leak-Free) | Train-Only Cancer Neighbors + Shared Functions | 0.2458 | 0.3150 | 0.8465 | 0.3869 | 0.3333 | 0.4609 |
| Centrality-Driven Graph Fusion (XGBoost) | Centrality + Functional + Topology + GNN Emb | 0.2363 | 0.2949 | 0.8355 | 0.3686 | 0.3357 | 0.4087 |
| Random Forest (Centrality) | 11 Centrality Metrics | 0.2255 | 0.2398 | 0.7789 | 0.2659 | 0.1991 | 0.4000 |
| XGBoost (Centrality) | 11 Centrality Metrics | 0.2085 | 0.2158 | 0.7869 | 0.2723 | 0.2667 | 0.2783 |
| Logistic Regression (Centrality) | 11 Centrality Metrics | 0.2173 | 0.2029 | 0.8019 | 0.2741 | 0.2387 | 0.3217 |
| SVM-RBF (Centrality) | 11 Centrality Metrics | 0.1811 | 0.1813 | 0.7973 | 0.2468 | 0.2417 | 0.2522 |

### 3. 5-Way Feature Ablation Study (Unified XGBoost, Held-Out Test Split)

| Ablation | Features | Test PR-AUC | Test ROC-AUC | Test F1 |
| :--- | :---: | :---: | :---: | :---: |
| **A. Centrality-Only** | 11 | 0.2158 | 0.7869 | 0.2723 |
| **B. Functional-Only** (leak-scrubbed) | 35 | 0.1601 | 0.7643 | 0.2380 |
| **C. Centrality + Functional** | 46 | 0.3151 | 0.8526 | 0.3319 |
| **D. Graph Topology & Spectral Embeddings Only** | 51 | 0.2757 | 0.8153 | 0.3454 |
| **E. Combined Centrality-Driven Graph Model** | 113 | **0.3377** | 0.8391 | **0.3682** |

### 4. SHAP Explainability & Biological Validation
- **Feature Group Share of Total |SHAP| (`shap.TreeExplainer`):** Graph Neighborhood / Topology (**43.1%**), Functional Annotation (**31.2%**), Centrality (**25.7%**).
- **Top Individual Drivers:** `closeness_centrality` (`+0.759` corr), `harmonic_closeness` (`+0.690`), `log_protein_size` (`+0.841`), `func_svd_8` (`+0.901`), `nbr_std_log_degree` (`+0.758`), and `clustering_coefficient` (**`-0.840` negative corr** — cancer hubs bridge distinct signaling modules rather than sitting inside dense housekeeping cliques such as ribosomes).
- **Top-50 Model Enrichment:** `40 / 50 (80.0%)` known COSMIC cancer genes (vs `29 / 50` by centrality alone). Top-10 novel (non-COSMIC) candidates include `SP1`, `MAPK8`, `FOS`, `PGR`, `IGF1R`, `YY1`, `PRKCZ`, `SHC1`, `NFKB1`, and `ATF2`.

> **Scientific Disclaimer:** High cancer association probabilities prioritize candidate proteins for biological/experimental follow-up and do **not** constitute proof of cancer causation.

---

## Project Structure

```text
cancer-ppi-prediction/
├── config/                    # Dataset YAML configurations (STRING+COSMIC & Original Schema)
├── data/
│   ├── raw/                   # Optional original dataset directory
│   ├── processed/             # centrality_features.csv, ppi_edges.csv, proteins_metadata.csv, SHAP JSON, SQLite DB
│   └── external/              # DATASET_PROVENANCE.md (licensing & download instructions)
├── models/
│   ├── random_forest/         # Saved Random Forest checkpoints (.joblib)
│   ├── xgboost/               # Saved XGBoost checkpoints (.joblib)
│   ├── gcn/                   # Saved PyTorch GCN, GAT, GraphSAGE & Fusion-GNN weights (.pt)
│   └── saved_models/          # Preprocessing pipeline, calibrators, LR, SVM checkpoints
├── src/                       # Modular Python pipeline (data_loader -> explainability -> prediction)
├── backend/                   # FastAPI REST API + SQLite query persistence
├── frontend/                  # React 18 + Plotly.js 7-page interactive web application
├── results/
│   ├── figures/               # Publication-ready PNG plots (degree, centralities, ROC/PR, ablation, SHAP)
│   ├── tables/                # model_comparison.csv, ablation_comparison.csv, feature_importance.csv
│   └── predictions/           # protein_predictions.csv, cancer_protein_rankings.csv, evaluation_metrics.json, network_statistics.json
├── tests/                     # Automated pytest unit, anti-leakage, artifact & FastAPI integration tests
├── main.py                    # Unified CLI entry point (train, predict, serve)
├── requirements.txt
└── README.md
```

---

## Installation & Execution Commands

### 1. Install Dependencies
```bash
cd cancer-ppi-prediction
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline (Data -> Graph -> Centrality -> Models/GNNs -> Ranking -> SHAP)
```bash
python main.py
```
To run with an alternate dataset config (e.g., original schema files placed in `data/raw/original/`):
```bash
python main.py --config config/dataset_original_schema.yaml
```

### 3. Predict Specific Proteins from the Command Line
```bash
python main.py --predict TP53 BRCA1 SP1 9606.ENSP00000275493
```

### 4. Launch the Full-Stack Web Application (FastAPI + React + Plotly.js)
```bash
python main.py --serve --host 127.0.0.1 --port 8000
```
Then open **`http://127.0.0.1:8000`** in your browser to explore all 7 interactive views:
1. **Dashboard**
2. **Protein Prediction**
3. **Centrality Analysis** (with interactive composite weight sliders)
4. **PPI Network Graph** (interactive hub & ego-network explorer)
5. **Model Comparison** (ROC, PR, Confusion Matrices, 5-Way Ablation)
6. **Explainability (SHAP)** (global & per-protein local attributions)
7. **Dataset Information**

### 5. Run the Automated Verification Test Suite
```bash
pytest tests/test_pipeline.py -v
```

---

## Dataset Attribution
- **STRING v12.5 (*Homo sapiens*):** CC BY 4.0 — [https://string-db.org/](https://string-db.org/)
- **COSMIC Cancer Gene Census v104 (GRCh38):** Wellcome Sanger Institute Academic License — [https://cancer.sanger.ac.uk/census](https://cancer.sanger.ac.uk/census)
- **Reference Inspiration:** [`sevmardi/Protein-Protein-Networks`](https://github.com/sevmardi/Protein-Protein-Networks) (analyzed in Phase 1; original private dataset not redistributed).

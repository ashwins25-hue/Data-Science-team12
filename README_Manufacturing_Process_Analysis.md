# Explainable Manufacturing Process Pattern Analysis and Adaptive Risk Monitoring

## Project Overview

This data science micro-project analyzes process measurements from plastic injection molding used for automotive plastic components. It discovers manufacturing patterns using clustering, profiles those patterns against recorded quality classes, assigns illustrative risk levels, explains cluster characteristics, and demonstrates monitoring of new production records and process drift.

The notebook uses the plastic injection molding dataset linked in its code:

- Dataset repository: https://github.com/airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding
- Dataset file expected by the notebook: `verified_real_world_plastic_injection_dataset_extract.csv` (downloaded from the repository's raw-content URL in the notebook).

## Objectives

- Load and inspect manufacturing process data.
- Identify the quality-label column and map its coded values to readable quality categories.
- Clean numeric process features and handle missing values.
- Explore feature statistics and correlations.
- Reduce dimensionality using Principal Component Analysis (PCA).
- Compare K-Means, Gaussian Mixture Model (GMM), and DBSCAN clustering.
- Profile discovered clusters using quality and defect rates.
- Assign illustrative Low, Medium, and High risk levels to process clusters.
- Explain cluster characteristics using feature deviations and a Random Forest surrogate model.
- Evaluate high-risk defect alerts on held-out records.
- Demonstrate new-record monitoring, process drift detection, and engineer feedback logging.
- Export model artifacts, analysis tables, and monitoring results.

## Algorithms and Techniques

| Technique | Purpose in the notebook |
|---|---|
| Median imputation | Fill missing numerical feature values |
| StandardScaler | Standardize process measurements |
| PCA | Reduce the feature space while retaining 95% of explained variance for clustering |
| K-Means | Discover groups of similar production records |
| Gaussian Mixture Model (GMM) | Discover probabilistic clusters |
| DBSCAN | Compare density-based clusters and identify noise points |
| Silhouette score | Primary metric used to compare valid clustering solutions |
| Davies–Bouldin index | Additional clustering quality metric |
| Calinski–Harabasz score | Additional clustering quality metric |
| Random Forest surrogate | Explain which features help reproduce cluster assignments |
| Kolmogorov–Smirnov (KS) test | Detect distribution changes between reference and current process data |
| Bonferroni correction | Adjust the drift-test significance threshold for multiple features |

## Dataset and Quality Labels

The notebook searches for a quality column named `quality`, `quality_class`, or `label` (case-insensitive). It expects the source dataset's numeric quality codes to be:

| Code | Label |
|---:|---|
| 1 | Waste |
| 2 | Acceptable |
| 3 | Target |
| 4 | Inefficient |

Rows without a valid quality value are removed. The notebook defines `Defect = 1` for the `Waste` class and `0` for all other classes. The notebook then excludes the quality column and derived label fields from the process-feature set.

The code converts candidate process features to numeric values, removes features with more than 40% missing values, removes constant features, and uses median imputation for remaining missing values.

**Dataset availability:** The notebook attempts to download the CSV automatically. If that fails, it prompts for a manual CSV upload in Google Colab. Use the source dataset and verify its column names and label coding before running the analysis.

## Requirements

The notebook installs or imports these Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `scipy`

Install them in a local environment with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy
```

The notebook uses `google.colab.files` for fallback upload and final ZIP download, so **Google Colab is the intended execution environment**. To run locally in Jupyter, replace those Colab-specific upload/download steps and adjust the output directory from `/content/manufacturing_results` to a local path.

## How to Run

1. Open `DS_MP.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. Allow the notebook to install the required libraries.
4. Confirm the dataset downloads successfully, or upload the expected CSV when prompted.
5. Review the dataset shape, quality distribution, missing-value checks, feature statistics, and correlation plot.
6. Review PCA and clustering comparisons.
7. Inspect the selected clustering model, cluster profiles, risk levels, and feature explanations.
8. Review held-out defect-alert metrics and the confusion matrix.
9. Review the new-production-record monitoring example and process-drift results.
10. Download the generated `manufacturing_results.zip` archive.

## Workflow

1. **Load data:** Download or upload the injection-molding dataset.
2. **Identify quality labels:** Map codes to `Waste`, `Acceptable`, `Target`, and `Inefficient`.
3. **Clean features:** Convert process columns to numeric, remove excessively incomplete or constant features, and impute remaining missing values.
4. **Split records:** Create a stratified 75% training and 25% test split.
5. **Scale data:** Fit the imputer and standard scaler on the training split, then transform the test split.
6. **Explore data:** Produce summary statistics and a feature-correlation heatmap.
7. **Reduce dimensions:** Fit PCA to retain 95% of the training data's explained variance.
8. **Compare clusters:** Test K-Means and GMM with 2–8 clusters and DBSCAN with several `eps` settings.
9. **Select model:** Choose the highest-Silhouette K-Means/GMM model for deployment; DBSCAN remains a comparison model because it does not provide a native prediction method in this workflow.
10. **Profile risk:** Compare cluster defect and quality-category rates, then assign illustrative risk levels.
11. **Explain patterns:** Identify process features with the largest standardized cluster deviations and train a Random Forest surrogate to estimate feature importance for cluster membership.
12. **Evaluate alerts:** Use held-out records to calculate precision, recall, F1, average precision (PR-AUC in the notebook's label), and a confusion matrix.
13. **Monitor process:** Classify one held-out record as an example of a new production record.
14. **Detect drift:** Compare reference and current feature distributions using KS tests with Bonferroni correction; also test an artificially shifted feature as a demonstration.
15. **Capture feedback:** Record an example engineer decision. The feedback is logged but does not automatically retrain the model.
16. **Export outputs:** Save CSV reports and a serialized model bundle, then package them into a ZIP file.

## Risk Interpretation

The notebook computes each cluster's defect rate from training data. It labels a cluster:

- **HIGH:** cluster defect rate is at least twice the overall training defect rate.
- **MEDIUM:** cluster defect rate is at least the overall training defect rate, but below the High threshold.
- **LOW:** cluster defect rate is below the overall training defect rate.

These are prototype thresholds. They must be validated with manufacturing and quality engineers before operational use. A cluster's historical defect rate is an estimate based on the data assigned to that cluster, not a guarantee about a future product.

## Process Drift Detection

The notebook compares each standardized feature between the training reference data and a current dataset using the two-sample Kolmogorov–Smirnov test. It uses a Bonferroni-adjusted threshold (`alpha / number_of_features`) to reduce false positives from testing many features.

The notebook also adds an artificial shift of `+2.0` standardized units to the first test feature to demonstrate how a distribution change may be detected. This is a simulation, not evidence of a real factory event.

A detected shift indicates a distribution change; it does not by itself prove defects or identify the cause.

## Generated Outputs

The notebook saves these files inside `/content/manufacturing_results/` and bundles them into `manufacturing_results.zip`:

| Output | Contents |
|---|---|
| `train_cluster_results.csv` | Training records with cluster assignments and derived quality/risk fields |
| `test_risk_predictions.csv` | Held-out records, cluster assignments, risk labels, and defect-risk estimates |
| `cluster_risk_profile.csv` | Per-cluster sample counts, defect rates, inefficient-category rates, risk lift, and risk level |
| `cluster_explanations.csv` | Top process features deviating from the overall training mean for each cluster |
| `feature_importance.csv` | Random Forest surrogate feature importance for cluster membership |
| `drift_results.csv` | KS statistics, p-values, and drift flags for the reference-versus-test comparison |
| `human_feedback.csv` | Example engineer feedback log |
| `manufacturing_model.joblib` | Saved imputer, scaler, PCA transformer, clustering model, surrogate model, feature list, risk map, defect-rate map, and selected model name |
| `manufacturing_results.zip` | ZIP archive containing the saved output files |

## Limitations and Responsible Use

- Clustering discovers similarity patterns; it does not directly establish causal reasons for defects.
- The Random Forest surrogate explains cluster assignments, not proven defect causes.
- The risk thresholds are illustrative and require validation.
- The held-out evaluation depends on the dataset, quality coding, cluster assignments, and selected model.
- The notebook's simulated drift test is artificial and should not be reported as a real manufacturing incident.
- The example engineer feedback is demonstration data, not an actual engineer assessment.
- The saved feedback is not used for automatic model retraining.
- Before production deployment, validate the workflow on independent production data, review performance across product and machine types, and involve manufacturing, process, and quality engineers.

## Suggested Future Enhancements

- Validate cluster stability across time periods, machines, and production batches.
- Establish risk thresholds with quality engineers and documented acceptance criteria.
- Add a production dashboard with alerts, trend history, and audit logs.
- Test drift detection on genuinely time-ordered production data.
- Develop a reviewed feedback-to-retraining process with versioning and approval.
- Add explainability that distinguishes correlation from causal evidence.
- Evaluate the system prospectively before using it to guide factory interventions.

## Suggested Repository Structure

```text
Manufacturing-Process-Analysis/
├── DS_MP.ipynb
├── README.md
└── manufacturing_results.zip   # generated after a successful run
```

The source dataset is downloaded by the notebook when available, so it may not need to be stored in the repository. If you include it, follow the dataset repository's license and usage terms.

## License

No project license is specified in the notebook. Check the dataset repository's license and add an appropriate license for your own project before public distribution.

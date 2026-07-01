# mRS Outcome Prediction and Hemorrhage Segmentation Toolkit

This repository contains the code used for CT hemorrhage segmentation, radiomics feature extraction, XGBoost-based mRS outcome prediction, model fusion, and calibration evaluation.
the model file has been uploaded to Google Cloud Drive (https://drive.google.com/drive/folders/1WLX-wvq2OqfL2ngBpIzPaEXx3t-zRGUf?usp=sharing).

The workflow has four main stages:

1. Train or run a 3D U-Net hemorrhage segmentation model.
2. Extract preoperative, postoperative, and delta radiomics features.
3. Train and evaluate M1-M4 XGBoost outcome models plus the final stacking fusion model.
4. Compute additional calibration metrics such as Brier scores.

## Repository Files

| File | Purpose |
| --- | --- |
| `train.py` | Trains a 3D U-Net segmentation model and supports validation-only evaluation. |
| `batch_inference.py` | Runs batch segmentation inference on `.nii.gz` CT files. |
| `extract_radiomics_features.py` | Extracts radiomics, basic volume/HU statistics, delta features, and derived clinical features. |
| `radiomics_config.yaml` | PyRadiomics feature extraction configuration. |
| `xgboost_mrs_analysis.py` | Trains and evaluates M1-M4 XGBoost mRS prediction models and the final stacking fusion model. |
| `calculate_brier_scores.py` | Computes Brier scores from prediction probability tables. |
| `M1.joblib` | Saved clinical model bundle. |
| `M2.joblib` | Saved preoperative radiomics model bundle. |
| `M3.joblib` | Saved postoperative radiomics model bundle. |
| `M4.joblib` | Saved delta radiomics model bundle. |
| `Fusion_stack_M1_M2_M3_M4.joblib` | Saved final stacking fusion model using M1-M4 probabilities. |

## Environment

The scripts are written for Python 3 and require common scientific Python packages.

Core dependencies:

```bash
numpy
pandas
scipy
scikit-learn
xgboost
statsmodels
matplotlib
seaborn
shap
torch
SimpleITK
pyradiomics
cc3d
batchgenerators
tqdm
openpyxl
PyYAML
```

GPU acceleration is optional for XGBoost and PyRadiomics, but strongly recommended for 3D U-Net training and inference.

## Expected Data Layout

For segmentation training and inference, CT image and segmentation files should use paired NIfTI names:

```text
patients/
  ID.nii.gz
  ID_seg.nii.gz
  ID-1.nii.gz
  ID-1_seg.nii.gz
```

Naming convention:

| Pattern | Meaning |
| --- | --- |
| `ID.nii.gz` | Preoperative image |
| `ID_seg.nii.gz` | Preoperative segmentation |
| `ID-1.nii.gz` | Postoperative image |
| `ID-1_seg.nii.gz` | Postoperative segmentation |

Segmentation labels:

| Label | Structure |
| --- | --- |
| `1` | IVH |
| `2` | SAH |
| `3` |residual ventricle |

For the mRS prediction pipeline, the radiomics feature CSV files must contain:

```text
patient_id
mRS
```

`mRS_binary` is generated automatically:

```text
mRS 0-2 -> 0
mRS 3-6 -> 1
```

## 1. Segmentation Training

Edit the configuration block at the bottom of `train.py`:

```python
config["run_mode"] = "train"
config["data_dir"] = "patients"
config["output_dir"] = "training_output"
config["feature_csv_path"] = "features_train_val.csv"
config["target_size"] = (182, 218, 182)
```

Run:

```bash
python train.py
```

Main outputs:

```text
training_output/
  config.json
  training_log.csv
  eval_report_epoch*.xlsx
  logs/
  checkpoints/
    latest.pth
    best.pth
```

The current segmentation preprocessing uses:

```python
target_size = (182, 218, 182)
small_hemorrhage_ml = 0.5
voxel_volume_mm3 = 1.0
```

With `voxel_volume_mm3 = 1.0`, the 0.5 ml threshold corresponds to 500 voxels.

## 2. Validation-Only Segmentation Evaluation

To evaluate a saved checkpoint, edit the bottom of `train.py`:

```python
config["run_mode"] = "val_only"
config["resume_from"] = "training_output/checkpoints/best.pth"
config["save_val_predictions"] = True
```

Run:

```bash
python train.py
```

Validation outputs include:

```text
training_output/
  val_eval_detailed.xlsx
  val_predictions_manifest.csv
  val_predictions/
```

## 3. Batch Segmentation Inference

`batch_inference.py` loads trained checkpoints and exports predicted segmentation masks.

Expected local layout:

```text
input_images/
  ID.nii.gz
  ID-1.nii.gz
expertA.pth
expertB.pth
batch_inference.py
train.py
```

Edit the configuration block:

```python
DEVICE = "cuda"
BASE_CHANNELS = 32
TARGET_SIZE = (182, 218, 182)
THRESHOLD = 0.5
POSTPROCESS = False
MIN_SIZE = 0
```

Run:

```bash
python batch_inference.py
```

Predicted files are saved as:

```text
ID_ExpertA_seg.nii.gz
ID_ExpertB_seg.nii.gz
```

## 4. Radiomics Feature Extraction

`extract_radiomics_features.py` extracts features from paired pre/post CT images and masks.

Default upload-mode paths:

```text
data/output_efy/
data/output_th/
radiomics_output/
```

Run:

```bash
python extract_radiomics_features.py
```

Main outputs:

```text
data/featuresefy.csv
data/featuresth.csv
radiomics_output/efy/radiomics_features.csv
radiomics_output/th/radiomics_features.csv
```

The script extracts:

- Basic volume and HU statistics.
- Label-specific radiomics for IVH, SAH, and Ventricle.
- Total ventricular features from IVH plus Ventricle.
- Delta features between preoperative and postoperative scans.
- Derived clinical features such as clearance rate and ventricular change indices.

PyRadiomics settings are controlled by:

```text
radiomics_config.yaml
```

## 5. XGBoost mRS Analysis

`xgboost_mrs_analysis.py` trains four base models:

| Model | Feature group |
| --- | --- |
| M1 | Clinical features |
| M2 | Preoperative radiomics |
| M3 | Postoperative radiomics |
| M4 | Delta radiomics |

It also evaluates decision-level stacking fusion models and saves the final full fusion model:

```text
Fusion_stack_M1_M2_M3_M4.joblib
```

Default upload-mode layout:

```text
data/
  featuresyjs.csv
  featurestl.csv
  featuresfy.csv
  featuresay.csv
  featuresth.csv
  featuresefy.csv
xgboost_mrs_results/
```

Run the full pipeline:

```bash
python xgboost_mrs_analysis.py
```

Main outputs:

```text
xgboost_mrs_results/
  table_model_performance.csv
  table_model_performance_ci.csv
  table_final_model_features.csv
  table_youden_thresholds_and_confusion.csv
  table_test_center_delong.csv
  table_hl_calibration.csv
  prediction_probabilities.csv
  mrs_xgboost_analysis_tables.xlsx
  trained_models/
    M1.joblib
    M2.joblib
    M3.joblib
    M4.joblib
    Fusion_stack_M1_M2_M3_M4.joblib
    manifest.csv
```

## 6. Test-Only XGBoost Mode

The XGBoost script also supports a test-only mode that reuses fixed features and saved model bundles.

Edit:

```python
ONLY_TEST_MODE = True
```

Required files:

```text
xgboost_mrs_results/table_final_model_features.csv
xgboost_mrs_results/trained_models/M1.joblib
xgboost_mrs_results/trained_models/M2.joblib
xgboost_mrs_results/trained_models/M3.joblib
xgboost_mrs_results/trained_models/M4.joblib
```

Run:

```bash
python xgboost_mrs_analysis.py
```

Test-only outputs are written under:

```text
xgboost_mrs_results/test_only/
```

## 7. Brier Score Calculation

After running the XGBoost pipeline, compute Brier scores:

```bash
python calculate_brier_scores.py
```

Input:

```text
xgboost_mrs_results/prediction_probabilities.csv
```

Output:

```text
xgboost_mrs_results/table_brier_scores.csv
```

## Saved Model Bundles

The `.joblib` files are model bundles, not plain estimators.

Base model bundles contain:

```text
model_name
model
columns
imputer
threshold
random_state
```

The final fusion model bundle contains:

```text
model_name
model_type
model
input_models
threshold
random_state
```

For `Fusion_stack_M1_M2_M3_M4.joblib`, `input_models` is:

```text
["M1", "M2", "M3", "M4"]
```

To use the fusion model, first generate probabilities from M1-M4, stack them in that order, then call:

```python
fusion_model.predict_proba(stacked_probability_matrix)[:, 1]
```


## Typical End-to-End Order

```bash
python train.py
python batch_inference.py
python extract_radiomics_features.py
python xgboost_mrs_analysis.py
python calculate_brier_scores.py
```

Use only the steps needed for your experiment. For example, if segmentation masks and feature CSV files are already available, start from `xgboost_mrs_analysis.py`.

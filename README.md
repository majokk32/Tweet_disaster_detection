# Tweet_disaster_detection
A projetct in Kaggle

# Step 1 to Step 3 README

This section of the repository covers the earlier project notebooks:

- `Group17_Step1_Step2.ipynb`
- `Group17_Step3_Transformers.ipynb`

These notebooks build the shared processed dataset, run the TF-IDF baselines, and produce the first round of transformer fine-tuning results.

## Notebook Location and Paths

These notebooks were developed in **Google Colab** with the project files placed in a shared Google Drive folder.

Shared Google Drive project folder:

https://drive.google.com/drive/u/0/folders/1eVBiAIEmZzN_E4e6qNbQrPPjqjbVhdFr

When the notebooks are executed, Google Drive is mounted at `/content/drive` and the project root used by the cells is:

```text
/content/drive/MyDrive/CS544-Group17-Project/
```

## What Each Notebook Does

### `Group17_Step1_Step2.ipynb`

This notebook covers:

1. **Data loading** of the raw Kaggle files (`train.csv`, `test.csv`)
2. **Exploratory Data Analysis** including shape checks, label distribution, and duplicate detection
3. **Preprocessing pipeline** that produces the shared processed CSVs used by all later notebooks
4. **TF-IDF Baselines** with Logistic Regression, Linear SVM, and Multinomial Naive Bayes
5. **Results summary table** comparing the three baseline models

The processed files written here are reused by every later step:

- `train_processed.csv`
- `test_processed.csv`

### `Group17_Step3_Transformers.ipynb`

This notebook covers the first round of transformer fine-tuning:

1. **DistilBERT** fine-tuning
2. **RoBERTa** fine-tuning
3. **BERTweet** fine-tuning
4. **Metadata ablation** on BERTweet, comparing text only vs. text + keyword vs. text + keyword + location

This notebook depends on the processed CSVs produced by Step 1/2.

## Environment Setup

Both notebooks were prepared for **Google Colab** with a GPU runtime.

Step 1/2 uses standard scientific Python libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`) that are pre-installed in Colab.

Step 3 additionally installs the HuggingFace stack inside the notebook:

```python
!pip install transformers datasets -q
```

Both notebooks mount Google Drive before any data cell runs:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Device and System Used

These notebooks were developed and run on the following setup:

- Platform: **Google Colab**
- Step 1/2 runs on the default Colab CPU runtime
- Step 3 runs on a Colab **GPU** runtime (CUDA is required for transformer fine-tuning)

## Expected Data Layout

The notebooks assume the shared Google Drive folder is organized as:

```text
/content/drive/MyDrive/CS544-Group17-Project/
├── input/
│   ├── train.csv
│   └── test.csv
└── data/
    ├── train_processed.csv     # produced by Step 1/2
    └── test_processed.csv      # produced by Step 1/2
```

The raw Kaggle files live under `input/`.
The processed files written by Step 1/2 live under `data/` and are read by Step 3.

In Step 1/2, the directory variable points at the raw Kaggle inputs:

```python
DATA_DIR = '/content/drive/MyDrive/CS544-Group17-Project/input/'
```

In Step 3, the directory variables are split between processed and raw:

```python
DATA_DIR  = '/content/drive/MyDrive/CS544-Group17-Project/data/'
INPUT_DIR = '/content/drive/MyDrive/CS544-Group17-Project/input/'
```

## How to Run the Code

Run the notebooks in this order:

1. `Group17_Step1_Step2.ipynb`
2. `Group17_Step3_Transformers.ipynb`

Each notebook is designed to be executed with **Run All**.

The order matters because Step 3 reads the `train_processed.csv` and `test_processed.csv` files written by Step 1/2.

## How the Results Are Generated

### Step 1/2 Results

Step 1/2 first runs the preprocessing pipeline on the raw Kaggle data and writes the cleaned CSVs back to Google Drive at `/content/drive/MyDrive/CS544-Group17-Project/data/`.

It then evaluates three TF-IDF + classical model combinations with stratified cross-validation:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Multinomial Naive Bayes

The strongest baseline is retrained on the full training set, and a Kaggle submission file is produced in the notebook working directory:

```text
submission_baseline.csv
```

### Step 3 Results

Step 3 fine-tunes three transformer backbones on the processed data:

- DistilBERT
- RoBERTa
- BERTweet

It also runs the BERTweet metadata ablation, comparing:

- text only
- text + keyword
- text + keyword + location

The selected transformer model produces a Kaggle submission file in the notebook working directory:

```text
submission_transformer.csv
```

## Summary

These two notebooks form the earlier foundation of the project:

- Step 1/2 produces the shared processed dataset and the classical-baseline reference point.
- Step 3 produces the first transformer results and motivates the backbone choices used in the later Step 6 to Step 9 pipeline.

Together they generate the processed inputs and the early baseline scores that every subsequent notebook depends on.



# Step 3.2 Next Steps README

This section documents the notebook:

- `Group17_Step3_2_NextSteps_fixed.ipynb`

This notebook extends the Step 3 transformer work.
It tests practical next-step improvements around threshold tuning, soft voting, class weighting, and error analysis.

## Notebook Location and Paths

The notebook is expected to live in the **repository root**:

- `./Group17_Step3_2_NextSteps_fixed.ipynb`

The notebook was prepared in **Google Colab** with the project files placed in the shared Google Drive project folder:

```text
/content/drive/MyDrive/CS544-Group17-Project/
```

When executed, Google Drive is mounted at `/content/drive`, and the project root used by the cells is:

```python
PROJECT_DIR = '/content/drive/MyDrive/CS544-Group17-Project/'
INPUT_DIR = PROJECT_DIR + 'input/'
DATA_DIR = PROJECT_DIR + 'data/'
NEW_OUTPUT_DIR = PROJECT_DIR + 'step3_2_next_steps_artifacts/'
```

## What This Notebook Does

### `Group17_Step3_2_NextSteps_fixed.ipynb`

This notebook covers:

1. **Threshold tuning** for the TF-IDF Logistic Regression baseline
2. **BERTweet fine-tuning** using `vinai/bertweet-base`
3. **Threshold tuning** for the BERTweet validation probabilities
4. **Soft-voting ensemble search** between Logistic Regression and BERTweet
5. **Class weight adjustment** to test whether minority-class recall improves
6. **Error analysis** comparing baseline-only errors, transformer-only errors, and shared errors

The notebook is designed as a lightweight follow-up to Step 3.

## Environment Setup

The notebook was prepared for **Google Colab** with a GPU runtime.

The optional install cell is included near the top of the notebook:

```python
# !pip install -q transformers datasets sentencepiece emoji==0.6.0
```

The main Python libraries used are:

- `pandas`
- `numpy`
- `torch`
- `scikit-learn`
- `transformers`
- `datasets`
- `sentencepiece`
- `emoji`

The notebook also mounts Google Drive before loading project data:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Device and System Used

This notebook was developed and run on the following setup:

- Platform: **Google Colab**
- Runtime: **GPU**
- Colab GPU type recorded in notebook metadata: **T4**
- Model backbone: `vinai/bertweet-base`

CUDA is strongly recommended because the notebook fine-tunes BERTweet twice:

- default training
- class-weighted training

## Expected Data Layout

The notebook assumes the processed files from Step 1/2 already exist:

```text
/content/drive/MyDrive/CS544-Group17-Project/
├── input/
│   ├── train.csv
│   └── test.csv
└── data/
    ├── train_processed.csv
    └── test_processed.csv
```

The required input paths are:

```text
/content/drive/MyDrive/CS544-Group17-Project/data/train_processed.csv
/content/drive/MyDrive/CS544-Group17-Project/data/test_processed.csv
```

The notebook creates its output folder automatically:

```text
/content/drive/MyDrive/CS544-Group17-Project/step3_2_next_steps_artifacts/
```

## How to Run the Code

Run this notebook after Step 1/2 and the first Step 3 transformer notebook:

1. `Group17_Step1_Step2.ipynb`
2. `Group17_Step3_Transformers.ipynb`
3. `Group17_Step3_2_NextSteps_fixed.ipynb`

The notebook is designed to be executed with **Run All**.

The order matters because Step 3.2 reads the processed CSV files written by Step 1/2 and builds on the baseline and BERTweet findings from Step 3.

## How the Results Are Generated

### Baseline Threshold Tuning

The notebook trains a TF-IDF + Logistic Regression baseline on an 80/20 stratified split of the processed training data.
It then sweeps thresholds from 0.05 to 0.95 and selects the threshold with the best validation F1 score.

The best recorded Logistic Regression result is:

- Threshold: `0.43`
- F1: `0.754177`
- Precision: `0.765751`
- Recall: `0.742947`
- Accuracy: `0.794137`

### BERTweet Threshold Tuning

The notebook fine-tunes `vinai/bertweet-base` for 3 epochs on the same split.
It then tunes the classification threshold on validation probabilities.

The best recorded BERTweet default result is:

- Threshold: `0.41`
- F1: `0.809562`
- Precision: `0.823339`
- Recall: `0.796238`
- Accuracy: `0.840773`

### Soft-Voting Ensemble

The notebook searches ensemble weights between the Logistic Regression probabilities and the BERTweet probabilities.
The best recorded ensemble is:

```text
ensemble = 0.95 * BERTweet + 0.05 * LogisticRegression
```

with:

- Threshold: `0.53`
- F1: `0.810373`
- Precision: `0.838926`
- Recall: `0.783699`
- Accuracy: `0.844104`

### Class Weight Adjustment

The notebook reruns BERTweet with balanced class weights:

```text
class 0 weight = 0.870107
class 1 weight = 1.175480
```

The class-weighted run did not improve over the default BERTweet run on this validation split:

- Default BERTweet F1: `0.809562`
- Class-weighted BERTweet F1: `0.805423`

### Error Analysis

The notebook uses the default BERTweet model for deeper validation-set error analysis.
The recorded comparison is:

- both correct: `1116`
- both wrong: `163`
- baseline only wrong: `146`
- transformer only wrong: `76`

This analysis helps identify where the transformer improves over the classical baseline and where both model families still fail.

## Expected Outputs

Main files written to `step3_2_next_steps_artifacts/`:

- `baseline_threshold_curve.csv`
- `baseline_threshold_summary.json`
- `transformer_default_history.csv`
- `transformer_default_threshold_curve.csv`
- `transformer_class_weight_history.csv`
- `transformer_class_weight_threshold_curve.csv`
- `class_weight_comparison.csv`
- `ensemble_weight_threshold_search.csv`
- `submission_ensemble_soft_vote.csv`
- `error_comparison_summary.csv`
- `error_cases_baseline_only_wrong.csv`
- `error_cases_transformer_only_wrong.csv`
- `error_cases_both_wrong.csv`
- `baseline_false_positives_val.csv`
- `baseline_false_negatives_val.csv`
- `transformer_false_positives_val.csv`
- `transformer_false_negatives_val.csv`
- `step3_2_next_steps_summary.json`

## Summary

This notebook confirms that most of the Step 3.2 improvement comes from BERTweet and threshold-aware probability use.
The Logistic Regression baseline remains useful for comparison and error analysis, but contributes only a small weight in the best soft-voting ensemble.

The Step 3.2 results bridge the earlier Step 3 transformer experiments and the later Step 6 to Step 9 fusion pipeline by showing that threshold tuning, OOF-style probability thinking, and model complementarity are worth pursuing further.


# Step 4 to Step 5 README

This section documents the notebooks:

- `step4_metadata_ablation_artifacts.ipynb`
- `step5_metadata_injection.ipynb`

These notebooks cover the metadata-focused part of the project. Step 4 packages additional midterm artifacts and reruns the BERTweet metadata ablation. Step 5 then tests whether `keyword` and `location` metadata become more useful when they are injected through more structured modeling strategies.

The official Step 5 results described here are from the local `nlp` environment run, which is the version aligned with the submitted final report.

## Notebook Location and Paths

Both notebooks are expected to live in the **repository root**:

- `./step4_metadata_ablation_artifacts.ipynb`
- `./step5_metadata_injection.ipynb`

### Step 4 Paths

Step 4 was prepared in **Google Colab** with the project files placed in the shared Google Drive project folder:

```text
/content/drive/MyDrive/CS544-Group17-Project/
```

When executed, Google Drive is mounted at `/content/drive`, and the project root used by the cells is:

```python
PROJECT_DIR = '/content/drive/MyDrive/CS544-Group17-Project/'
INPUT_DIR = PROJECT_DIR + 'input/'
DATA_DIR = PROJECT_DIR + 'data/'
NEW_OUTPUT_DIR = PROJECT_DIR + 'step4_midterm_artifacts/'
```

### Step 5 Paths

Step 5 was prepared for a local repository checkout. When the notebook is executed from the repository root, the path variables resolve to:

```python
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / 'data'
STEP5_DIR = PROJECT_DIR / 'step5_metadata_injection'
OUTPUT_DIR = STEP5_DIR / 'step5_outputs'
RUNS_DIR = OUTPUT_DIR / 'runs'
```

The notebook stores every completed run/fold under:

```text
./step5_metadata_injection/step5_outputs/runs/{run_id}/fold_{k}/
```

## What Each Notebook Does

### `step4_metadata_ablation_artifacts.ipynb`

This notebook packages the additional Step 4 artifacts prepared for TA review. It is designed as a clean Colab runbook that leaves the earlier Step 1 to Step 3 notebooks unchanged.

It covers:

1. **Colab setup** with Google Drive mounting and transformer dependency installation
2. **Shared path setup** for raw inputs, processed data, and Step 4 artifacts
3. **Baseline artifact generation** for TF-IDF classical models
4. **Validation diagnostics** including confusion matrix and false-positive/false-negative files
5. **BERTweet metadata ablation** comparing text only, text + keyword, and text + keyword + location
6. **Artifact verification** to confirm every expected output file exists
7. **Summary preview** of the key CSV and JSON outputs

### `step5_metadata_injection.ipynb`

This notebook runs the main metadata-injection study. It asks whether the Kaggle metadata fields can improve BERTweet beyond the Step 4 text-only anchor when the metadata is injected more carefully.

It covers:

1. **Metadata cleaning** for `location`, `country_norm`, and junk-location flags
2. **Pre-EDA diagnostics** for metadata coverage and keyword/text overlap
3. **A 12-run experiment matrix** built from structured `RunSpec` objects
4. **Method A: tokenizer-pair metadata input**
5. **Method B: natural-language metadata templates**
6. **Method C: late-fusion metadata head**
7. **Fold-local vocabularies** to avoid validation leakage
8. **5-fold stratified cross-validation** with cached fold-level metrics
9. **Paired delta analysis** against the `anchor_text` baseline
10. **Report-ready tables and figures** for the final writeup

## Environment Setup

### Step 4 Environment

Step 4 was prepared for **Google Colab** with a GPU runtime.

The notebook installs the required transformer packages near the top:

```python
pip install transformers==4.46.3 datasets sentencepiece emoji==0.6.0
```

The main Python libraries used are:

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `torch`
- `transformers`
- `datasets`
- `sentencepiece`
- `emoji`

The notebook also mounts Google Drive before loading project data:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5 Environment

Step 5 was prepared for a local Conda environment named `nlp`.

Create and activate the environment:

```bash
conda create -n nlp python=3.10 -y
conda activate nlp
```

Install the main packages:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 pandas==2.3.3 scikit-learn==1.7.2 matplotlib==3.10.8
pip install sentencepiece==0.2.1 emoji==0.6.0 datasets==4.8.4 accelerate==1.13.0
pip install notebook==7.5.5 jupyterlab==4.5.6 ipywidgets==8.1.8
```

Core package versions used for the official Step 5 run:

```text
python              3.10.20
torch               2.4.1+cu121
transformers        4.46.3
pandas              2.3.3
scikit-learn        1.7.2
matplotlib          3.10.8
sentencepiece       0.2.1
emoji               0.6.0
datasets            4.8.4
accelerate          1.13.0
notebook            7.5.5
jupyterlab          4.5.6
ipywidgets          8.1.8
```

The Step 5 notebook sets `MPLCONFIGDIR` inside `step5_metadata_injection/step5_outputs/.matplotlib` so matplotlib does not write to a system-level config directory.

## Device and System Used

### Step 4 Device

Step 4 was developed and run on the following setup:

- Platform: **Google Colab**
- Runtime: **GPU**
- Colab GPU type recorded in notebook metadata: **H100**
- Python recorded in notebook metadata: **3.12**
- Model backbone: `vinai/bertweet-base`

### Step 5 Device

Step 5 was developed and run on the following setup:

- Operating system: local Linux workstation
- Conda environment: `nlp`
- Python: `3.10.20`
- PyTorch: `2.4.1+cu121`
- GPU: `NVIDIA RTX 4070 Ti`
- Device reported by notebook: `cuda`
- Model backbone: `vinai/bertweet-base`
- Random seed: `42`

CUDA is required for the full Step 5 matrix because it fine-tunes BERTweet for 12 runs across 5 folds.

## Expected Data Layout

### Step 4 Data Layout

Step 4 assumes the shared Google Drive folder is organized as:

```text
/content/drive/MyDrive/CS544-Group17-Project/
|-- input/
|   |-- train.csv
|   `-- test.csv
|-- data/
|   |-- train_processed.csv
|   `-- test_processed.csv
`-- step4_midterm_artifacts/
```

If the processed files already exist under `data/`, the notebook loads them. Otherwise, it rebuilds the required processed files from the raw Kaggle inputs under `input/`.

### Step 5 Data Layout

Step 5 assumes the processed files from the earlier notebooks have been generated or copied into the local repository checkout:

```text
./data/train_processed.csv
./data/test_processed.csv
```

These files are produced by `Group17_Step1_Step2.ipynb`. If the `data/` folder is not present after cloning the repository, run the earlier preprocessing notebook first or copy the two processed CSVs into `./data/` before opening Step 5.

The notebook creates its output directory automatically:

```text
./step5_metadata_injection/step5_outputs/
```

## How to Run the Code

Run Step 4 after the Step 1 to Step 3 processed data has been created:

1. `Group17_Step1_Step2.ipynb`
2. `Group17_Step3_Transformers.ipynb`
3. `step4_metadata_ablation_artifacts.ipynb`

Run Step 5 after Step 4 or after confirming that the processed CSVs exist locally:

4. `step5_metadata_injection.ipynb`

### Running Step 4

1. Open `step4_metadata_ablation_artifacts.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run the Colab setup cell and mount Google Drive when prompted.
4. Confirm that the shared project folder exists at:

```text
/content/drive/MyDrive/CS544-Group17-Project/
```

5. Run the notebook with **Run All**.
6. Confirm that the output verification section prints `[OK]` for every expected file.

### Running Step 5

1. Activate the local environment:

```bash
conda activate nlp
```

2. Start Jupyter:

```bash
jupyter lab
```

3. Open `step5_metadata_injection.ipynb` from the repository root.
4. Confirm that `DATA_DIR` points to `./data/`.
5. Run the setup, data loading, metadata cleaning, run specification, and function-definition cells.
6. Use the full-matrix switches:

```python
RUN_PRE_EDA = True
RUN_SMOKE = False
RUN_FULL_MATRIX = True
RUN_CONDITIONAL_EXPERIMENTS = False
RESUME = True
FORCE_RERUN = False
FORCE_RUN_IDS = []
```

7. Run the full matrix section only when CUDA is available.
8. Run the cache rebuild and report-ready sections after training. These sections regenerate tables and figures from cached fold results without retraining.

The notebook is resumable. If training is interrupted, rerun with `RESUME=True`; completed folds with matching config hashes are skipped.

## How the Results Are Generated

### Step 4 Results

Step 4 first regenerates the baseline review artifacts. The baseline pipeline uses cleaned tweet text, TF-IDF features, and 5-fold stratified cross-validation to compare:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

Logistic Regression is then used for validation diagnostics, including the classification report, confusion matrix, false-positive examples, false-negative examples, and top weighted TF-IDF features.

Step 4 then trains and evaluates three BERTweet metadata-ablation settings:

- `bertweet_text`
- `bertweet_text_keyword`
- `bertweet_text_keyword_location`

The Step 4 BERTweet settings are:

- Seed: `42`
- Epochs: `3`
- Batch size: `16`
- Learning rate: `2e-5`
- Model: `vinai/bertweet-base`

The key Step 4 ablation results are:

```text
BERTweet text + keyword + location: F1 = 0.8078
BERTweet text only:                F1 = 0.8063
BERTweet text + keyword:           F1 = 0.8060
```

The metadata improvement in Step 4 was very small, which motivated the more careful Step 5 metadata-injection study.

### Step 5 Results

Step 5 uses `vinai/bertweet-base`, 5-fold stratified cross-validation, seed `42`, 3 epochs, batch size `16`, and `F1_pos` as the primary metric. `F1_pos` is the binary F1 score for the disaster class (`target=1`).

The full Step 5 matrix contains 12 planned runs:

```text
anchor_text      text-only BERTweet anchor
A_kw             tokenizer-pair input with keyword
A_kw_loc         tokenizer-pair input with keyword and cleaned location
B_kw             natural-language metadata template with keyword
B_kw_loc         natural-language metadata template with keyword and cleaned location
C_head_only      late-fusion head control without active metadata embeddings
C_kw             late-fusion keyword feature
C_kw_loc         late-fusion keyword and cleaned-location features
C_kw_loc_raw     late-fusion keyword and raw-location-derived features
C_loc            late-fusion location-only feature
C_kw_mask        late-fusion keyword/location with keyword masked from text
A_kw_country     tokenizer-pair input with keyword and normalized country
```

For every `run_id` and fold, the notebook writes validation predictions, metrics, and a configuration hash. The aggregation stage rebuilds:

- per-run cross-validation summaries
- per-fold metrics
- paired fold deltas against `anchor_text`
- report-ready tables, figures, and markdown snippets

The official local Step 5 matrix found no reliable metadata improvement:

```text
anchor_text F1_pos = 0.8061
best contender B_kw F1_pos = 0.8065
delta vs anchor = +0.0004
positive folds = 2 / 5
decision = noise
```

The Step 5 conclusion is that the available `keyword` and `location` metadata did not produce a consistent practical gain over the text-only BERTweet anchor under the tested injection strategies.

## Expected Outputs

### Step 4 Outputs

Step 4 writes the following files to `step4_midterm_artifacts/`:

- `baseline_cv_summary.csv`
- `baseline_validation_report.json`
- `baseline_confusion_matrix.csv`
- `baseline_false_positives.csv`
- `baseline_false_negatives.csv`
- `baseline_top_positive_features.csv`
- `baseline_top_negative_features.csv`
- `transformer_cv_summary.csv`
- `bertweet_ablation_summary.csv`
- `transformer_fold_metrics.csv`
- `transformer_run_metadata.json`
- `submission_transformer.csv`

### Step 5 Outputs

Step 5 writes fold-level outputs under `step5_metadata_injection/step5_outputs/runs/{run_id}/fold_{k}/`:

- `metrics.json`
- `val_preds.csv`
- `config_hash.txt`

It also writes summary and report-ready outputs:

- `step5_pre_eda.json`
- `step5_run_metadata.json`
- `step5_cv_summary.csv`
- `step5_fold_metrics.csv`
- `step5_paired_deltas.csv`
- `report_ready/step5_report_main_summary.csv`
- `report_ready/step5_report_paired_summary.csv`
- `report_ready/step5_report_q_table.csv`
- `report_ready/step5_report_delta_matrix.csv`
- `report_ready/step5_f1_pos_comparison.png`
- `report_ready/step5_paired_delta_heatmap.png`
- `report_ready/step5_report_snippets.md`

## Summary

Step 4 confirms that simple BERTweet metadata concatenation gives only a very small improvement over text-only BERTweet. Step 5 tests more structured metadata-injection methods and finds that none produce a reliable practical gain over the text-only anchor in the official local run.

Together, these notebooks support the final report's conclusion that later improvements should come from stronger backbones, threshold-aware probability use, and model fusion rather than additional metadata engineering alone.

-------



# Step 6 to Step 9 README

This repository contains the final four notebooks used in the Step 6 to Step 9 phase of the project:

- `step6_backbone_refresh.ipynb`
- `step7_screen_fusion.ipynb`
- `step8_textonly_refresh.ipynb`
- `step9_hybrid_fusion.ipynb`

These notebooks document the final modeling path that led to our best clean submission.
The pipeline focuses on transformer training, out-of-fold prediction generation, and probability-level fusion.

## Notebook Location and Paths

The four notebooks are expected to live in the **repository root**:

- `./step6_backbone_refresh.ipynb`
- `./step7_screen_fusion.ipynb`
- `./step8_textonly_refresh.ipynb`
- `./step9_hybrid_fusion.ipynb`

When the notebooks are executed, the repository root is treated as the working directory.

Example local project root:

```text
D:/CS544-Group17-Project/
```

Example Colab-style project root used earlier in development:

```text
/content/drive/MyDrive/CS544-Group17-Project/
```

## What Each Notebook Does

### `step6_backbone_refresh.ipynb`

This notebook reruns the main transformer backbone comparison under a clean local setup.
It keeps the two models that matter for the later stages:

- `BERTweet` with `text + keyword pair`
- `Twitter-RoBERTa` with `text + keyword pair`

The goal of Step 6 is to produce:

- 5-fold cross-validation metrics
- out-of-fold (OOF) probabilities
- test probabilities
- threshold tuning summaries

### `step7_screen_fusion.ipynb`

This notebook fuses the two useful Step 6 models.
It searches over fusion weights and decision thresholds using OOF predictions.

The Step 7 idea is simple:

- keep the two strongest keyword-aware models
- average their probabilities with different weights
- choose the best threshold on OOF validation predictions

### `step8_textonly_refresh.ipynb`

This notebook tests whether a metadata-free text-only branch can add complementary signal.
It compares several text-only runs and keeps the strongest one for the final fusion stage.

The main purpose of Step 8 is not to replace Step 7 directly, but to provide a second view of the data.

### `step9_hybrid_fusion.ipynb`

This notebook builds the final submission model.
It fuses:

- the best Step 7 keyword-aware ensemble
- the best Step 8 text-only model

Like Step 7, it uses OOF-based weight search and threshold tuning.

## Environment Setup

All notebooks were prepared for a Conda environment named `usc-nlp`.

Create and activate the environment:

```bash
conda create -n usc-nlp python=3.12 -y
conda activate usc-nlp
```

Install the main packages:

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install pandas==3.0.1 numpy==2.4.2 scikit-learn==1.8.0 transformers==5.3.0
pip install ipython jupyter
```

## Device and System Used

The notebooks were developed and run on the following setup:

- Operating system: `Windows 11`
- Python: `3.12.12`
- Conda environment: `usc-nlp`
- PyTorch: `2.6.0+cu124`
- CUDA: `12.4`
- GPU: `NVIDIA GeForce RTX 4070 SUPER`

All transformer experiments in Step 6 to Step 9 were intended to run on **CUDA**, not CPU.

## Expected Data Layout

The notebooks assume the repository contains:

- `data/train_processed.csv`
- `data/test_processed.csv`

These files are used as the shared inputs across the four notebooks.

In other words, the expected input paths are:

```text
./data/train_processed.csv
./data/test_processed.csv
```

## How to Run the Code

Run the notebooks in this order:

1. `step6_backbone_refresh.ipynb`
2. `step7_screen_fusion.ipynb`
3. `step8_textonly_refresh.ipynb`
4. `step9_hybrid_fusion.ipynb`

Each notebook is designed to be executed with **Run All**.

The order matters because:

- Step 7 depends on Step 6 outputs
- Step 9 depends on both Step 7 and Step 8 outputs

Each notebook writes its results to a stage-specific output folder under the repository root:

- Step 6 -> `./step6_outputs/`
- Step 7 -> `./step7_outputs/`
- Step 8 -> `./step8_outputs/`
- Step 9 -> `./step9_outputs/`

## How the Results Are Generated

### Step 6 Results

Step 6 trains the backbone models with **5-fold cross-validation**.
Each fold produces validation predictions, and those predictions are combined into OOF probabilities.

These OOF predictions are important because they are later used by the fusion notebooks.

### Step 7 Results

Step 7 does not train a new transformer from scratch.
Instead, it loads the Step 6 OOF probabilities and searches for the best probability blend.

The selected Step 7 fusion is:

```text
s71 = 0.69 * s61 + 0.31 * s62
```

This stage improved over the individual Step 6 models and became the best keyword-aware result.

### Step 8 Results

Step 8 removes metadata and tests a pure text-only branch.
The most useful Step 8 run is the longer BERTweet text-only model, which provides a different prediction bias from Step 7.

This branch is valuable mainly because it contributes diversity for the final fusion.

### Step 9 Results

Step 9 combines the strongest Step 7 and Step 8 outputs.
The final hybrid fusion is:

```text
s91 = 0.65 * s71 + 0.35 * s84
```

The threshold is then tuned on OOF predictions to produce the final classification rule.

## Final Result

The best final result from this Step 6 to Step 9 pipeline was:

- OOF F1: `0.817020`
- Kaggle public leaderboard score: `0.84707`
- Kaggle public leaderboard rank: `20`

## Summary

The main finding from these four notebooks is that the strongest improvements came from:

- using a strong tweet-domain backbone
- generating reliable OOF predictions
- combining complementary models through soft fusion

The final submission path is therefore a **two-stage soft-voting ensemble**, not a single-model solution.

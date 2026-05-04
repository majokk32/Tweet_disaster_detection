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

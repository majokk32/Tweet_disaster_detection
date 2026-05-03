# Tweet_disaster_detection
A projetct in Kaggle





Step 6 to Step 9 README
This repository contains the final four notebooks used in the Step 6 to Step 9 phase of the project:

step6_backbone_refresh.ipynb
step7_screen_fusion.ipynb
step8_textonly_refresh.ipynb
step9_hybrid_fusion.ipynb
These notebooks document the final modeling path that led to our best clean submission.
The pipeline focuses on transformer training, out-of-fold prediction generation, and probability-level fusion.

Notebook Location and Paths
The four notebooks are expected to live in the repository root:

./step6_backbone_refresh.ipynb
./step7_screen_fusion.ipynb
./step8_textonly_refresh.ipynb
./step9_hybrid_fusion.ipynb
When the notebooks are executed, the repository root is treated as the working directory.

Example local project root:

D:/CS544-Group17-Project/
Example Colab-style project root used earlier in development:

/content/drive/MyDrive/CS544-Group17-Project/
What Each Notebook Does
step6_backbone_refresh.ipynb
This notebook reruns the main transformer backbone comparison under a clean local setup.
It keeps the two models that matter for the later stages:

BERTweet with text + keyword pair
Twitter-RoBERTa with text + keyword pair
The goal of Step 6 is to produce:

5-fold cross-validation metrics
out-of-fold (OOF) probabilities
test probabilities
threshold tuning summaries
step7_screen_fusion.ipynb
This notebook fuses the two useful Step 6 models.
It searches over fusion weights and decision thresholds using OOF predictions.

The Step 7 idea is simple:

keep the two strongest keyword-aware models
average their probabilities with different weights
choose the best threshold on OOF validation predictions
step8_textonly_refresh.ipynb
This notebook tests whether a metadata-free text-only branch can add complementary signal.
It compares several text-only runs and keeps the strongest one for the final fusion stage.

The main purpose of Step 8 is not to replace Step 7 directly, but to provide a second view of the data.

step9_hybrid_fusion.ipynb
This notebook builds the final submission model.
It fuses:

the best Step 7 keyword-aware ensemble
the best Step 8 text-only model
Like Step 7, it uses OOF-based weight search and threshold tuning.

Environment Setup
All notebooks were prepared for a Conda environment named usc-nlp.

Create and activate the environment:

conda create -n usc-nlp python=3.12 -y
conda activate usc-nlp
Install the main packages:

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install pandas==3.0.1 numpy==2.4.2 scikit-learn==1.8.0 transformers==5.3.0
pip install ipython jupyter
Device and System Used
The notebooks were developed and run on the following setup:

Operating system: Windows 11
Python: 3.12.12
Conda environment: usc-nlp
PyTorch: 2.6.0+cu124
CUDA: 12.4
GPU: NVIDIA GeForce RTX 4070 SUPER
All transformer experiments in Step 6 to Step 9 were intended to run on CUDA, not CPU.

Expected Data Layout
The notebooks assume the repository contains:

data/train_processed.csv
data/test_processed.csv
These files are used as the shared inputs across the four notebooks.

In other words, the expected input paths are:

./data/train_processed.csv
./data/test_processed.csv
How to Run the Code
Run the notebooks in this order:

step6_backbone_refresh.ipynb
step7_screen_fusion.ipynb
step8_textonly_refresh.ipynb
step9_hybrid_fusion.ipynb
Each notebook is designed to be executed with Run All.

The order matters because:

Step 7 depends on Step 6 outputs
Step 9 depends on both Step 7 and Step 8 outputs
Each notebook writes its results to a stage-specific output folder under the repository root:

Step 6 -> ./step6_outputs/
Step 7 -> ./step7_outputs/
Step 8 -> ./step8_outputs/
Step 9 -> ./step9_outputs/
How the Results Are Generated
Step 6 Results
Step 6 trains the backbone models with 5-fold cross-validation.
Each fold produces validation predictions, and those predictions are combined into OOF probabilities.

These OOF predictions are important because they are later used by the fusion notebooks.

Step 7 Results
Step 7 does not train a new transformer from scratch.
Instead, it loads the Step 6 OOF probabilities and searches for the best probability blend.

The selected Step 7 fusion is:

s71 = 0.69 * s61 + 0.31 * s62
This stage improved over the individual Step 6 models and became the best keyword-aware result.

Step 8 Results
Step 8 removes metadata and tests a pure text-only branch.
The most useful Step 8 run is the longer BERTweet text-only model, which provides a different prediction bias from Step 7.

This branch is valuable mainly because it contributes diversity for the final fusion.

Step 9 Results
Step 9 combines the strongest Step 7 and Step 8 outputs.
The final hybrid fusion is:

s91 = 0.65 * s71 + 0.35 * s84
The threshold is then tuned on OOF predictions to produce the final classification rule.

Final Result
The best final result from this Step 6 to Step 9 pipeline was:

OOF F1: 0.817020
Kaggle public leaderboard score: 0.84707
Kaggle public leaderboard rank: 20
Summary
The main finding from these four notebooks is that the strongest improvements came from:

using a strong tweet-domain backbone
generating reliable OOF predictions
combining complementary models through soft fusion
The final submission path is therefore a two-stage soft-voting ensemble, not a single-model solution.

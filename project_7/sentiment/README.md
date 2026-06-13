# Sentiment Analysis — Project 7 — Group 19

Binary sentiment classification on Amazon product reviews  
(Bittlingmayer dataset, Kaggle).

## Project structure

```
sentiment/
├── train.py                  # main entry point
├── requirements.txt
├── data/                     # train.ft.txt and test.ft.txt here
├── out/                      # models and plots are saved here
└── src/
    ├── data/
    │   ├── loader.py         # load & split the fastText dataset
    │   └── preprocessor.py  # base text cleaning
    ├── models/
    │   ├── logistic.py       # TF-IDF + Logistic Regression
    │   ├── lstm.py           # LSTM (PyTorch)
    │   └── roberta.py        # RoBERTa fine-tuning
    └── evaluation/
        └── metrics.py        # accuracy, F1, confusion matrix
```

## Setup

```bash
pip install -r sentiment/requirements.txt
```

## Data

the dataset is taken from Kaggle:  
https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

```
data/
├── train.ft.txt
└── test.ft.txt
```

## Training

### Logistic Regression (fast baseline, ~5 min on full data)

```bash
python3 sentiment/train.py \
  --model logistic \
  --train sentiment/data/train.ft.txt \
  --test  sentiment/data/test.ft.txt \
  --output_dir sentiment/out/logistic
```

### BiLSTM (requires PyTorch, ~1h on CPU / ~15 min on GPU)

```bash
python3 sentiment/train.py \
  --model lstm \
  --train sentiment/data/train.ft.txt \
  --test  sentiment/data/test.ft.txt \
  --device cuda \
  --output_dir sentiment/out/lstm
```

### RoBERTa (requires GPU, ~2–3h)

```bash
python3 sentiment/train.py \
  --model roberta \
  --train sentiment/data/train.ft.txt \
  --test  sentiment/data/test.ft.txt \
  --device cuda \
  --output_dir sentiment/out/roberta
```

## Quick test on a small subset

Add `--max_train 10000 --max_test 2000` to any command to run a fast
sanity check before training on the full dataset.

```bash
python3 sentiment/train.py --model logistic \
  --train sentiment/data/train.ft.txt --test sentiment/data/test.ft.txt \
  --max_train 10000 --max_test 2000
```

## Output

Each run saves:
- The trained model (`.pkl` for logistic, `.pt` for LSTM, directory for RoBERTa)
- A confusion matrix PNG in `out/<model>/`
- A `metrics.json` file with hyperparameters, runtime, validation metrics, and test metrics
- A `training_history.json` file for neural models
- Accuracy, Macro F1, Precision, Recall printed to stdout

To aggregate all experiment metrics into a CSV table:

```bash
python3 sentiment/aggregate_results.py \
  --root sentiment/out/experiments \
  --output sentiment/out/experiments/results.csv
```

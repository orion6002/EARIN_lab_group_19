# Sentiment Analysis — Project 7 — Group 19

Binary sentiment classification on Amazon product reviews  
(Bittlingmayer dataset, Kaggle).

## Project structure

```
sentiment/
├── train.py                  # main entry point
├── requirements.txt
├── data/                     # put train.ft.txt and test.ft.txt here
├── out/                      # models and plots are saved here
└── src/
    ├── data/
    │   ├── loader.py         # load & split the fastText dataset
    │   └── preprocessor.py  # base text cleaning
    ├── models/
    │   ├── logistic.py       # TF-IDF + Logistic Regression
    │   ├── lstm.py           # Bidirectional LSTM (PyTorch)
    │   └── roberta.py        # RoBERTa fine-tuning (HuggingFace)
    └── evaluation/
        └── metrics.py        # accuracy, F1, confusion matrix
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the dataset from Kaggle:  
https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

Extract the `.bz2` files and place them in `data/`:

```
data/
├── train.ft.txt
└── test.ft.txt
```

## Training

### Logistic Regression (fast baseline, ~5 min on full data)

```bash
python train.py \
  --model logistic \
  --train data/train.ft.txt \
  --test  data/test.ft.txt \
  --output_dir out/logistic
```

### BiLSTM (requires PyTorch, ~1h on CPU / ~15 min on GPU)

```bash
python train.py \
  --model lstm \
  --train data/train.ft.txt \
  --test  data/test.ft.txt \
  --device cuda \
  --output_dir out/lstm
```

### RoBERTa (requires GPU, ~2–3h)

```bash
python train.py \
  --model roberta \
  --train data/train.ft.txt \
  --test  data/test.ft.txt \
  --device cuda \
  --output_dir out/roberta
```

## Quick test on a small subset

Add `--max_train 10000 --max_test 2000` to any command to run a fast
sanity check before training on the full dataset.

```bash
python train.py --model logistic \
  --train data/train.ft.txt --test data/test.ft.txt \
  --max_train 10000 --max_test 2000
```

## Output

Each run saves:
- The trained model (`.pkl` for logistic, `.pt` for LSTM, directory for RoBERTa)
- A confusion matrix PNG in `out/<model>/`
- Accuracy, Macro F1, Precision, Recall printed to stdout

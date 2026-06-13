"""
models/lstm.py
LSTM / Bi-LSTM classifier for sentiment analysis.

Configurable options
--------------------
  bidirectional   : True  -> BiLSTM  |  False -> classic LSTM
  num_layers      : stack N LSTM layers
  freeze_embedding: lock / unlock the embedding layer (first layer)
  freeze_classifier: lock / unlock the linear head (last layer)
  train_size      : use only N examples from the training set
  lr              : Adam learning rate
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Maps tokens to integer indices."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self) -> None:
        self.token2idx: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx2token: dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts: list[str], max_vocab: int = 50_000) -> None:
        """
        Build vocabulary from training texts.

        Args:
            texts: List of whitespace-tokenised strings.
            max_vocab: Maximum vocabulary size (most frequent tokens kept).
        """
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())

        for token, _ in counter.most_common(max_vocab - 2):   # -2 for PAD/UNK
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        print(f"Vocabulary built: {len(self.token2idx):,} tokens.")

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer indices."""
        unk = self.token2idx[self.UNK_TOKEN]
        return [self.token2idx.get(t, unk) for t in text.split()]

    def __len__(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# 2.  PyTorch Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Encodes and pads reviews for the LSTM."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_len: int = 256,
    ) -> None:
        self.labels = labels
        self.max_len = max_len
        self.encoded = [vocab.encode(t) for t in texts]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        seq = self.encoded[idx]
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))   # 0 = PAD index

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3.  LSTM / BiLSTM architecture
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    Configurable LSTM / BiLSTM for binary sentiment classification.

    Architecture:
        Embedding  ->  LSTM (N layers, optional bidirectional)
                   ->  Global Max Pool  ->  Dropout  ->  Linear (2)

    Args:
        vocab_size      : Number of tokens in the vocabulary.
        embed_dim       : Embedding dimension (default 100).
        hidden_dim      : Hidden units per LSTM direction (default 128).
        num_layers      : Number of stacked LSTM layers (default 1).
        bidirectional   : If True, use BiLSTM; if False, use classic LSTM.
        dropout         : Dropout probability (default 0.3).
        pad_idx         : Index of the padding token (default 0).
        pretrained_embeddings : Optional numpy array (vocab_size, embed_dim).
        freeze_embedding  : If True, embedding weights are NOT updated.
        freeze_classifier : If True, the final linear layer is NOT updated.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1, # might be changed to max 5 for test purposes
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embedding: bool = False,
        freeze_classifier: bool = False,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional

        # ---- Embedding layer (first layer) ----
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings).float()
            )
        self.embedding.weight.requires_grad = not freeze_embedding

        # ---- LSTM (N layers, classic or bidirectional) ----
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # inter-layer dropout only when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output size: *2 if bidirectional, *1 if classic
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # ---- Linear head (last layer) ----
        self.classifier = nn.Linear(lstm_output_dim, 2)
        for param in self.classifier.parameters():
            param.requires_grad = not freeze_classifier

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids : (batch, seq_len)
        embedded = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embed)
        output, _ = self.lstm(embedded)                      # (batch, seq_len, lstm_out)
        # Global max pooling over time dimension
        pooled, _ = output.max(dim=1)                        # (batch, lstm_out)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)                     # (batch, 2)
        return logits

    def print_frozen_status(self) -> None:
        """Print which layers are frozen / trainable."""
        print("\n--- Layer status ---")
        for name, param in self.named_parameters():
            status = "TRAINABLE" if param.requires_grad else "FROZEN"
            print(f"  {name:<45} {status}")
        print()


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------

def train(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    vocab: Vocabulary,
    # ---- architecture ----
    bidirectional: bool = True,
    num_layers: int = 1,
    max_len: int = 256,
    embed_dim: int = 100,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    pretrained_embeddings: Optional[np.ndarray] = None,
    # ---- layer freezing ----
    freeze_embedding: bool = False,
    freeze_classifier: bool = False,
    # ---- training ----
    batch_size: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    train_size: Optional[int] = None,   # limit training examples (None = all)
    seed: int = 42,
    num_workers: int = 0,
    device: str = "cpu",
) -> LSTMClassifier:
    """
    Train the LSTM / BiLSTM classifier.

    Key parameters
    --------------
    bidirectional   : True -> BiLSTM, False -> classic LSTM
    num_layers      : number of stacked LSTM layers
    freeze_embedding: freeze the embedding layer (first layer)
    freeze_classifier: freeze the linear head (last layer)
    train_size      : use only this many training examples (reproducible subset)
    lr              : Adam learning rate
    weight_decay    : L2 regularization used by Adam
    num_workers     : DataLoader workers; 0 is safest on macOS/sandboxed runs

    Returns
    -------
    Best LSTMClassifier checkpoint (by validation accuracy).
    """
    # ---- Build datasets ----
    full_train_ds = ReviewDataset(train_texts, train_labels, vocab, max_len)

    if train_size is not None and train_size < len(full_train_ds):
        indices = list(range(train_size))   # first N (already shuffled by loader.py)
        train_ds = Subset(full_train_ds, indices)
        print(f"Training on subset: {train_size:,} / {len(full_train_ds):,} examples.")
    else:
        train_ds = full_train_ds
        print(f"Training on full dataset: {len(train_ds):,} examples.")

    val_ds = ReviewDataset(val_texts, val_labels, vocab, max_len)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---- Build model ----
    mode = "BiLSTM" if bidirectional else "LSTM"
    print(
        f"\nBuilding {mode} | layers={num_layers} | hidden={hidden_dim} "
        f"| embed={embed_dim} | lr={lr} | dropout={dropout} "
        f"| weight_decay={weight_decay} | freeze_emb={freeze_embedding} "
        f"| freeze_clf={freeze_classifier}"
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embedding=freeze_embedding,
        freeze_classifier=freeze_classifier,
    ).to(device)

    model.print_frozen_status()

    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters left. Check frozen layer options.")
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(num_epochs):
        # ---- Training pass ----
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Validation pass ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["label"].to(device)
                logits    = model(input_ids)
                preds     = logits.argmax(dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        val_acc = correct / total
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_history = history
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# 5.  Inference
# ---------------------------------------------------------------------------

def predict(
    model: LSTMClassifier,
    texts: list[str],
    vocab: Vocabulary,
    max_len: int = 256,
    batch_size: int = 128,
    device: str = "cpu",
) -> list[int]:
    """
    Run inference on a list of preprocessed texts.

    Returns
    -------
    List of predicted integer labels.
    """
    dataset = ReviewDataset(texts, [0] * len(texts), vocab, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits    = model(input_ids)
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    return preds


# ---------------------------------------------------------------------------
# 6.  Save / Load
# ---------------------------------------------------------------------------

def save(model: LSTMClassifier, vocab: Vocabulary, path: str) -> None:
    """Save model weights and vocabulary to disk."""
    import pickle
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path + ".pt")
    with open(path + ".vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"Model saved to {path}.pt")


def load(path: str, device: str = "cpu") -> tuple[LSTMClassifier, Vocabulary]:
    """Load model weights and vocabulary from disk."""
    import pickle

    with open(path + ".vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    state = torch.load(path + ".pt", map_location=device)

    # Infer architecture from saved weights
    lstm_out_dim  = state["classifier.weight"].shape[1]
    embed_dim     = state["embedding.weight"].shape[1]
    vocab_size    = state["embedding.weight"].shape[0]
    # lstm.weight_hh_l0 shape: (4*hidden, hidden) for classic
    #                           (4*hidden, 2*hidden) for bidirectional
    hidden_dim    = state["lstm.weight_hh_l0"].shape[0] // 4
    bidirectional = (lstm_out_dim == hidden_dim * 2)
    num_layers    = sum(1 for k in state if k.startswith("lstm.weight_ih_l"))

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(state)
    return model, vocab

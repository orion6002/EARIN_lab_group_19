"""
models/lstm.py
Bidirectional LSTM classifier for sentiment analysis.

Pipeline:
  1. Tokenize text -> integer sequences
  2. Pad / truncate to fixed length
  3. Embed with pre-trained GloVe (or random init)
  4. BiLSTM -> global max-pool -> dropout -> linear head
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
            texts: List of tokenized strings (whitespace-separated).
            max_vocab: Maximum vocabulary size (most frequent tokens kept).
        """
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())

        for token, _ in counter.most_common(max_vocab - 2):   # -2 for PAD / UNK
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
        # Truncate or pad to max_len
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))   # 0 = PAD index

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3.  BiLSTM Architecture
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for binary sentiment classification.

    Architecture:
        Embedding -> BiLSTM -> Global Max Pool -> Dropout -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        embedded = self.dropout(self.embedding(input_ids))   # (batch, seq_len, embed)
        output, _ = self.lstm(embedded)                       # (batch, seq_len, 2*hidden)
        # Global max pooling over time dimension
        pooled, _ = output.max(dim=1)                         # (batch, 2*hidden)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)                      # (batch, 2)
        return logits


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------

def train(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    vocab: Vocabulary,
    max_len: int = 256,
    embed_dim: int = 100,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    batch_size: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    pretrained_embeddings: np.ndarray | None = None,
) -> BiLSTMClassifier:
    """
    Train the BiLSTM classifier with early stopping on validation F1.

    Args:
        train_texts / train_labels: Training data.
        val_texts / val_labels: Validation data.
        vocab: Fitted Vocabulary object.
        max_len: Maximum sequence length (tokens).
        embed_dim: Embedding dimension.
        hidden_dim: LSTM hidden units per direction.
        dropout: Dropout rate.
        batch_size: Mini-batch size.
        num_epochs: Maximum training epochs.
        lr: Adam learning rate.
        device: "cpu" or "cuda".
        pretrained_embeddings: Optional (vocab_size, embed_dim) numpy array.

    Returns:
        Best BiLSTMClassifier (by validation accuracy).
    """
    train_ds = ReviewDataset(train_texts, train_labels, vocab, max_len)
    val_ds = ReviewDataset(val_texts, val_labels, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# 5.  Inference
# ---------------------------------------------------------------------------

def predict(
    model: BiLSTMClassifier,
    texts: list[str],
    vocab: Vocabulary,
    max_len: int = 256,
    batch_size: int = 128,
    device: str = "cpu",
) -> list[int]:
    """
    Run inference on a list of preprocessed texts.

    Returns:
        List of predicted integer labels.
    """
    dataset = ReviewDataset(texts, [0] * len(texts), vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    return preds


# ---------------------------------------------------------------------------
# 6.  Save / Load
# ---------------------------------------------------------------------------

def save(model: BiLSTMClassifier, vocab: Vocabulary, path: str) -> None:
    """Save model weights and vocabulary."""
    import pickle
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path + ".pt")
    with open(path + ".vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"LSTM model saved to {path}.pt")


def load(path: str, device: str = "cpu") -> tuple[BiLSTMClassifier, Vocabulary]:
    """Load model weights and vocabulary from disk."""
    import pickle

    with open(path + ".vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    state = torch.load(path + ".pt", map_location=device)
    # Infer hidden_dim from saved weights
    hidden_dim = state["classifier.weight"].shape[1] // 2
    embed_dim = state["embedding.weight"].shape[1]
    vocab_size = state["embedding.weight"].shape[0]

    model = BiLSTMClassifier(vocab_size, embed_dim, hidden_dim).to(device)
    model.load_state_dict(state)
    return model, vocab

"""
models/roberta.py
Fine-tuning of roberta-base for binary sentiment classification.

Pipeline:
  1. Tokenize with RoBERTa BPE tokenizer (max 512 tokens)
  2. Fine-tune roberta-base + linear classification head
  3. Optimise with AdamW + linear warmup scheduler
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1.  Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Tokenises reviews for RoBERTa."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: RobertaTokenizerFast,
        max_len: int = 512,
    ) -> None:
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 2.  Training loop
# ---------------------------------------------------------------------------

def train(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    model_name: str = "roberta-base",
    max_len: int = 512,
    batch_size: int = 32,
    num_epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    device: str = "cpu",
) -> tuple[RobertaForSequenceClassification, RobertaTokenizerFast]:
    """
    Fine-tune roberta-base for binary classification.

    Args:
        train_texts / train_labels: Training data.
        val_texts / val_labels: Validation data.
        model_name: HuggingFace model identifier.
        max_len: Maximum tokenizer length (tokens).
        batch_size: Mini-batch size.
        num_epochs: Number of fine-tuning epochs.
        lr: Peak learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        warmup_ratio: Fraction of total steps used for warmup.
        device: "cpu" or "cuda".

    Returns:
        (best_model, tokenizer)
    """
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    print("Tokenising training data ...")
    train_ds = ReviewDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = ReviewDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3.  Inference
# ---------------------------------------------------------------------------

def predict(
    model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizerFast,
    texts: list[str],
    max_len: int = 512,
    batch_size: int = 64,
    device: str = "cpu",
) -> list[int]:
    """
    Run inference on a list of preprocessed texts.

    Returns:
        List of predicted integer labels.
    """
    dataset = ReviewDataset(texts, [0] * len(texts), tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())

    return preds


# ---------------------------------------------------------------------------
# 4.  Save / Load
# ---------------------------------------------------------------------------

def save(
    model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizerFast,
    path: str,
) -> None:
    """Save fine-tuned model and tokenizer to a directory."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"RoBERTa model saved to {path}/")


def load(
    path: str,
    device: str = "cpu",
) -> tuple[RobertaForSequenceClassification, RobertaTokenizerFast]:
    """Load fine-tuned model and tokenizer from a directory."""
    tokenizer = RobertaTokenizerFast.from_pretrained(path)
    model = RobertaForSequenceClassification.from_pretrained(path).to(device)
    return model, tokenizer

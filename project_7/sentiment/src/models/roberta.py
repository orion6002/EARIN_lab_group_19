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
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
        tokenizer,
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
    max_len: int = 512, # absolute max, can be devided by two to reduce computation by 4
    batch_size: int = 32,
    num_epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    device: str = "cpu",
    num_layers_to_freeze: int = 0,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple:
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
        seed: Random seed used by the training DataLoader.
        num_workers: DataLoader workers; 0 is safest on macOS/sandboxed runs.

    Returns:
        (best_model, tokenizer)
    """

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    base_model = getattr(model, "roberta", getattr(model, "base_model", None))
    if base_model is None:
        raise ValueError(f"Cannot find a base transformer module for {model_name}")

    if num_layers_to_freeze == -1:
        # We freeze everything exept the linear head
        for name, param in base_model.named_parameters():
            param.requires_grad = False
    elif num_layers_to_freeze > 0:
        # We freeze the Nth first layers (0 à 5 par exemple)
        if hasattr(base_model, "embeddings"):
            for name, param in base_model.embeddings.named_parameters():
                param.requires_grad = False
        available_layers = len(base_model.encoder.layer)
        layers_to_freeze = min(num_layers_to_freeze, available_layers)
        print(f"Freezing first {layers_to_freeze}/{available_layers} encoder layers")
        for i in range(layers_to_freeze):
            for name, param in base_model.encoder.layer[i].named_parameters():
                param.requires_grad = False
    else:
        print("Full Fine-Tuning")

    print("Tokenising training data ...")
    train_ds = ReviewDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = ReviewDataset(val_texts, val_labels, tokenizer, max_len)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_state = None
    history = []

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
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3.  Inference
# ---------------------------------------------------------------------------


def predict(
    model,
    tokenizer,
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


def save(model, tokenizer, path: str) -> None:
    """Save fine-tuned model and tokenizer to a directory."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"RoBERTa model saved to {path}/")


def load(path: str, device: str = "cpu") -> tuple:
    """Load fine-tuned model and tokenizer from a directory."""
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    return model, tokenizer

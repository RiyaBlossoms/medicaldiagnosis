"""Training loop with metrics graphs, checkpointing, and evaluation."""

import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report

from config import Config
from model import HybridMedicalModel
from dataset import build_dataloaders
from utils import set_seed, AverageMeter, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    loss_meter = AverageMeter()

    correct = 0
    total = 0

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)

        loss = criterion(logits, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_meter.update(loss.item(), imgs.size(0))

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    return loss_meter.avg, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    loss_meter = AverageMeter()

    all_preds = []
    all_labels = []

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)

        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), imgs.size(0))

        all_preds.extend(logits.argmax(1).cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return loss_meter.avg, accuracy, all_preds, all_labels


def run_training(cfg: Config):

    print("STARTING TRAINING...")

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("DEVICE:", device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print("DATASETS LOADED")

    model = HybridMedicalModel(cfg).to(device)

    print("MODEL LOADED")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=1e-6
    )

    best_val_loss = float("inf")

    patience_counter = 0

    # -------------------------
    # METRICS STORAGE
    # -------------------------

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # -------------------------
    # TRAINING LOOP
    # -------------------------

    for epoch in range(1, cfg.num_epochs + 1):

        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        # SAVE METRICS

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        train_accuracies.append(tr_acc)
        val_accuracies.append(val_acc)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # SAVE BEST MODEL

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                cfg
            )

            print(f"✓ Saved best checkpoint (val_loss={val_loss:.4f})")

        else:

            patience_counter += 1

            if patience_counter >= cfg.patience:

                print(f"Early stopping at epoch {epoch}")

                break

    # -------------------------
    # FINAL TEST EVALUATION
    # -------------------------

    ckpt = torch.load(
        f"{cfg.checkpoint_dir}/{cfg.disease}_best.pt",
        map_location=device
    )

    model.load_state_dict(ckpt["model_state"])

    _, test_acc, preds, labels = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"\nTEST ACCURACY: {test_acc:.4f}")

    print("\nCLASSIFICATION REPORT:\n")

    print(classification_report(labels, preds))

    # -------------------------
    # PLOT METRICS GRAPHS
    # -------------------------

    plt.figure(figsize=(12, 5))

    # LOSS GRAPH

    plt.subplot(1, 2, 1)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.legend()

    # ACCURACY GRAPH

    plt.subplot(1, 2, 2)

    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Accuracy Curve")

    plt.legend()

    plt.tight_layout()

    # SAVE GRAPH IMAGE

    plt.savefig("training_graphs.png")

    print("\nGRAPHS SAVED AS training_graphs.png")

    # SHOW GRAPHS

    plt.show()


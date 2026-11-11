"""Utility helpers: seeding, checkpointing, metrics."""
import os, random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n
        self.count += n; self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, val_loss, cfg):
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.disease}_best.pt")
    torch.save({
        "epoch":       epoch,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "disease":     cfg.disease,
    }, path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt

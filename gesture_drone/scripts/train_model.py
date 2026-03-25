"""
Train a 4-class hand gesture classifier using EfficientNet-B0 transfer learning.

Run from WSL2 with GPU:
    cd /home/kyo/Projects/MLxDrone_larp
    source venv/bin/activate
    python gesture_drone/scripts/train_model.py

All output is mirrored to gesture_drone/models/train.log.
A background thread writes [ALIVE] to the log every 30 s — use
    tail -f gesture_drone/models/train.log
in a second terminal to confirm training is still running.
"""

import logging
import threading
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATASET   = ROOT / "dataset_cropped"
TRAIN_DIR = DATASET / "train"
VAL_DIR   = DATASET / "val"
MODEL_DIR = ROOT / "models"
MODEL_PATH  = MODEL_DIR / "gesture_model.pt"
CURVES_PATH = MODEL_DIR / "training_curves.png"
LOG_PATH    = MODEL_DIR / "train.log"

# ── hyperparameters ───────────────────────────────────────────────────────
# Confirmed by benchmark_training.py on RTX 3070 (8 GB VRAM).
BATCH_SIZE    = 128   # 256 OOMs with full EfficientNet-B0 gradients
NUM_WORKERS   = 8     # 515 samples/sec vs 472 for 10 workers
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15
PHASE1_LR          = 1e-3
PHASE2_LR          = 1e-4
WEIGHT_DECAY       = 1e-4
USE_COMPILE        = False  # torch.compile requires nvcc — not available on WSL2
EARLY_STOP_PATIENCE = 3     # stop a phase if val_acc doesn't improve for this many epochs
ACCURACY_THRESHOLD  = 98.0  # stop immediately once val_acc hits this — lowered for 5-class model
                             # (unknown class is heterogeneous, 99%+ unrealistic)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── logging ───────────────────────────────────────────────────────────────
# Logs go to file with timestamps. Console output uses tqdm.write() so the
# progress bar is never corrupted. The heartbeat thread writes [ALIVE] to
# the log file every 30 s — tail -f the log to monitor from another terminal.

def _setup_logger() -> logging.Logger:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = _setup_logger()


def info(msg: str, *args):
    """Log to file AND print to terminal via tqdm.write (keeps progress bar intact)."""
    formatted = msg % args if args else msg
    log.info(formatted)
    tqdm.write(formatted)


class Heartbeat:
    """Writes [ALIVE] to the log file every `interval` seconds (file only, not stdout)."""

    def __init__(self, interval: int = 30):
        self.interval = interval
        self._msg     = "initialising"
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def set(self, msg: str):
        self._msg = msg

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            log.info("[ALIVE] %s", self._msg)


heartbeat = Heartbeat(interval=30)


# ── GPU memory helper ──────────────────────────────────────────────────────

def gpu_stats() -> str:
    if DEVICE.type != "cuda":
        return ""
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    rsvd  = torch.cuda.memory_reserved()  / 1024 ** 3
    return f"VRAM {alloc:.1f}/{rsvd:.1f} GB"


# ── data transforms ────────────────────────────────────────────────────────
# CPU workers do the minimum: decode JPEG → float tensor.
# Heavy augmentations (crop, jitter, erasing) run on the GPU per-batch,
# which is much faster than running them sample-by-sample on CPU workers.

# Resize to a uniform spatial size before collation — images in dataset_cropped/
# have varying dimensions (HaGRID 512px vs MediaPipe crops). Collation requires
# all tensors in a batch to be the same shape. GPU augmentation does the rest.
cpu_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

cpu_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Applied to each batch on GPU after .to(DEVICE)
gpu_train_augment = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
]).to(DEVICE)


# ── data loaders ──────────────────────────────────────────────────────────

def build_dataloaders():
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=cpu_train_transform)
    val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=cpu_val_transform)

    info("Training images  : %d", len(train_ds))
    info("Validation images: %d", len(val_ds))
    info("Classes          : %s", str(train_ds.classes))

    class_counts = Counter(train_ds.targets)
    num_classes  = len(train_ds.classes)
    for i, name in enumerate(train_ds.classes):
        info("  %s: %d", name, class_counts[i])

    sample_weights = [1.0 / class_counts[label] for label in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    total = sum(class_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )
    info("Loss weights     : %s", str([f"{w:.2f}" for w in class_weights.tolist()]))

    loader_kwargs = dict(
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_ds.classes, class_weights


# ── model ─────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, num_classes),
    )
    model = model.to(DEVICE)
    if USE_COMPILE and hasattr(torch, "compile"):
        info("Compiling model with torch.compile (first batch will be slow)...")
        model = torch.compile(model)
    return model


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ── training loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch_label: str):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    n_batches    = len(loader)
    t0           = time.perf_counter()

    pbar = tqdm(loader, desc=epoch_label, leave=False)
    for batch_idx, (images, labels) in enumerate(pbar, 1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # GPU augmentation — all heavy transforms happen here on the GPU
        images = gpu_train_augment(images)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

        # Keep heartbeat message fresh so tail -f always shows meaningful state
        elapsed = time.perf_counter() - t0
        eta     = (elapsed / batch_idx) * (n_batches - batch_idx) if batch_idx > 1 else 0
        heartbeat.set(
            f"{epoch_label} | batch {batch_idx}/{n_batches} "
            f"({100*batch_idx//n_batches}%) "
            f"acc={100.0*correct/total:.1f}% loss={loss.item():.4f} "
            f"eta={eta/60:.1f}m | {gpu_stats()}"
        )

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, class_names) -> tuple:
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    per_class_c  = Counter()
    per_class_t  = Counter()

    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with autocast(device_type="cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)
        for pred, true in zip(predicted, labels):
            per_class_t[true.item()] += 1
            if pred == true:
                per_class_c[true.item()] += 1

    parts = [
        f"{class_names[i]}={100.0*per_class_c[i]/per_class_t[i]:.0f}%"
        for i in range(len(class_names)) if per_class_t[i] > 0
    ]
    info("    Per-class val: %s", " | ".join(parts))
    return running_loss / total, 100.0 * correct / total


# ── phase runner ──────────────────────────────────────────────────────────

def run_phase(model, train_loader, val_loader, criterion, optimizer, scaler, scheduler,
              num_epochs, phase_name, history, best_state, class_names,
              use_early_stopping=True):
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        label = f"{phase_name} [{epoch}/{num_epochs}]"
        t0    = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, label,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, class_names)
        scheduler.step(val_loss)

        lr  = optimizer.param_groups[0]["lr"]
        dur = time.perf_counter() - t0
        info(
            "%s  train_loss=%.4f  train_acc=%.1f%%  "
            "val_loss=%.4f  val_acc=%.1f%%  lr=%.1e  time=%.0fs  %s",
            label, train_loss, train_acc, val_loss, val_acc, lr, dur, gpu_stats(),
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_state["best_val_acc"]:
            best_state["best_val_acc"] = val_acc
            best_state["best_epoch"]   = len(history["val_acc"])
            raw = getattr(model, "_orig_mod", model)
            best_state["state_dict"] = {k: v.cpu().clone()
                                        for k, v in raw.state_dict().items()}
            info("  ★ New best  val_acc=%.1f%%", val_acc)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if use_early_stopping:
                info("  No improvement for %d/%d epochs", epochs_without_improvement, EARLY_STOP_PATIENCE)
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    info("  Early stopping %s after %d epochs (patience=%d)",
                         phase_name, epoch, EARLY_STOP_PATIENCE)
                    break

        if use_early_stopping and val_acc >= ACCURACY_THRESHOLD:
            info("  Accuracy threshold reached (%.1f%% >= %.1f%%) — stopping %s",
                 val_acc, ACCURACY_THRESHOLD, phase_name)
            break


# ── save & plot ───────────────────────────────────────────────────────────

def save_model(state_dict, class_names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict, "class_names": class_names}, MODEL_PATH)
    info("Best model saved → %s", str(MODEL_PATH))


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("CrossEntropy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)
    info("Training curves saved → %s", str(CURVES_PATH))


# ── main ──────────────────────────────────────────────────────────────────

def main():
    heartbeat.start()

    info("=" * 60)
    info("GESTURE RECOGNITION MODEL TRAINING")
    info("=" * 60)
    info("Device     : %s", str(DEVICE))
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info("GPU        : %s  (%.1f GB VRAM)", props.name, props.total_memory / 1024**3)
        info("PyTorch    : %s  |  CUDA %s", torch.__version__, torch.version.cuda)
    info("AMP        : ON  |  compile: %s", "ON" if USE_COMPILE else "OFF")
    info("Batch size : %d  |  Workers: %d  |  GPU augmentation: ON", BATCH_SIZE, NUM_WORKERS)
    info("Log file   : %s  (tail -f to monitor)", str(LOG_PATH))
    info("-" * 60)

    train_loader, val_loader, class_names, class_weights = build_dataloaders()
    num_classes = len(class_names)
    info("")

    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scaler    = GradScaler()

    history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = {"best_val_acc": 0.0, "best_epoch": 0, "state_dict": None}

    # ── phase 1: frozen backbone ─────────────────────────────────────
    info("=" * 60)
    info("PHASE 1 — Training classifier head (backbone frozen)")
    info("=" * 60)
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    info("Trainable: %s / %s", f"{trainable:,}", f"{total_p:,}")
    info("")

    opt1  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    sch1  = optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=2, factor=0.5)
    t_all = time.time()
    run_phase(model, train_loader, val_loader, criterion, opt1, scaler, sch1,
              PHASE1_EPOCHS, "Phase1", history, best_state, class_names,
              use_early_stopping=False)

    # ── phase 2: full fine-tuning ────────────────────────────────────
    info("")
    info("=" * 60)
    info("PHASE 2 — Full fine-tuning (backbone unfrozen)")
    info("=" * 60)
    unfreeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info("Trainable: %s / %s", f"{trainable:,}", f"{total_p:,}")
    info("")

    opt2  = optim.Adam(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    sch2  = optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=3, factor=0.5)
    run_phase(model, train_loader, val_loader, criterion, opt2, scaler, sch2,
              PHASE2_EPOCHS, "Phase2", history, best_state, class_names,
              use_early_stopping=True)

    elapsed = time.time() - t_all
    info("")
    info("=" * 60)
    info("TRAINING COMPLETE  (%.0fs / %.1f min)", elapsed, elapsed / 60)
    info("Best val accuracy : %.1f%%  (epoch %d)",
         best_state["best_val_acc"], best_state["best_epoch"])
    info("=" * 60)

    save_model(best_state["state_dict"], class_names)
    plot_curves(history)
    heartbeat.stop()


if __name__ == "__main__":
    main()

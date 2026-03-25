"""
Benchmark DataLoader + GPU throughput to find optimal batch_size × num_workers.

Run from WSL2 with GPU:
    cd /home/kyo/Projects/MLxDrone_larp
    source venv/bin/activate
    python gesture_drone/scripts/benchmark_training.py

The script prints a ranked table of (batch_size, workers) → samples/sec.
Pick the config with the highest samples/sec that still fits in VRAM.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset_cropped" / "train"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

WARMUP_BATCHES  = 10   # discard first N batches (DataLoader cold start)
MEASURE_BATCHES = 40   # batches to measure after warmup

BASE_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def gpu_memory_used_gb() -> float:
    return torch.cuda.memory_reserved() / 1024 ** 3


def benchmark_loader_only(batch_size: int, num_workers: int) -> dict:
    """Measure raw DataLoader throughput without GPU work (CPU/IO bound)."""
    ds = datasets.ImageFolder(str(TRAIN_DIR), transform=BASE_TRANSFORM)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=True,
    )

    it = iter(loader)
    for _ in range(WARMUP_BATCHES):
        try:
            imgs, _ = next(it)
            imgs.to(DEVICE, non_blocking=True)
        except StopIteration:
            it = iter(loader)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for i in range(MEASURE_BATCHES):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, _ = next(it)
        imgs.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        total += imgs.size(0)

    elapsed = time.perf_counter() - t0
    return {"samples_per_sec": total / elapsed, "elapsed": elapsed}


def benchmark_full_pass(batch_size: int, num_workers: int, model: nn.Module,
                        criterion: nn.Module) -> dict:
    """Measure end-to-end forward + backward pass throughput."""
    ds = datasets.ImageFolder(str(TRAIN_DIR), transform=BASE_TRANSFORM)
    num_classes = len(ds.classes)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler()

    model.train()
    it = iter(loader)

    for _ in range(WARMUP_BATCHES):
        try:
            imgs, lbls = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, lbls = next(it)
        imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            loss = criterion(model(imgs), lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    t0    = time.perf_counter()
    total = 0
    peak_vram = 0.0

    for _ in range(MEASURE_BATCHES):
        try:
            imgs, lbls = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, lbls = next(it)
        imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            loss = criterion(model(imgs), lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

        total     += imgs.size(0)
        peak_vram  = max(peak_vram, gpu_memory_used_gb())

    elapsed = time.perf_counter() - t0
    return {
        "samples_per_sec": total / elapsed,
        "peak_vram_gb":    peak_vram,
        "elapsed":         elapsed,
    }


def main():
    if DEVICE.type != "cuda":
        print("No CUDA device found — nothing to benchmark.")
        return

    print("=" * 70)
    print("  TRAINING THROUGHPUT BENCHMARK")
    print("=" * 70)
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Data  : {TRAIN_DIR}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"VRAM  : {total_vram:.1f} GB total")
    print()

    # Build a single model instance shared across full-pass benchmarks
    ds_tmp     = datasets.ImageFolder(str(TRAIN_DIR), transform=BASE_TRANSFORM)
    num_classes = len(ds_tmp.classes)
    model       = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, num_classes))
    model       = model.to(DEVICE)
    criterion   = nn.CrossEntropyLoss()

    # ── Phase A: DataLoader-only scan (broad grid) ────────────────────────
    batch_sizes  = [64, 128, 256, 512]
    worker_counts = [2, 4, 6, 8, 10]

    print("─" * 70)
    print(f"  PHASE A  — DataLoader-only throughput  ({MEASURE_BATCHES} batches each)")
    print("─" * 70)
    print(f"  {'batch':>6}  {'workers':>7}  {'samples/sec':>12}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*12}")

    loader_results = []
    for bs in batch_sizes:
        for nw in worker_counts:
            try:
                r = benchmark_loader_only(bs, nw)
                loader_results.append((bs, nw, r["samples_per_sec"]))
                print(f"  {bs:>6}  {nw:>7}  {r['samples_per_sec']:>11.0f}")
            except RuntimeError as e:
                print(f"  {bs:>6}  {nw:>7}  OOM: {e}")

    loader_results.sort(key=lambda x: x[2], reverse=True)
    print()
    print(f"  Best loader config: batch={loader_results[0][0]}, "
          f"workers={loader_results[0][1]}, "
          f"throughput={loader_results[0][2]:.0f} samples/sec")

    # ── Phase B: Full forward+backward scan (narrower grid) ───────────────
    top_bs = sorted({r[0] for r in loader_results[:6]})
    top_nw = sorted({r[1] for r in loader_results[:6]})
    # Always include the clear winners
    top_bs = sorted(set(top_bs) | {loader_results[0][0]})
    top_nw = sorted(set(top_nw) | {loader_results[0][1]})

    print()
    print("─" * 70)
    print(f"  PHASE B  — Full forward+backward (AMP)  ({MEASURE_BATCHES} batches each)")
    print("─" * 70)
    print(f"  {'batch':>6}  {'workers':>7}  {'samples/sec':>12}  {'peak VRAM':>10}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*12}  {'─'*10}")

    full_results = []
    for bs in top_bs:
        for nw in top_nw:
            try:
                torch.cuda.reset_peak_memory_stats()
                r = benchmark_full_pass(bs, nw, model, criterion)
                full_results.append((bs, nw, r["samples_per_sec"], r["peak_vram_gb"]))
                vram_pct = 100.0 * r["peak_vram_gb"] / total_vram
                print(f"  {bs:>6}  {nw:>7}  {r['samples_per_sec']:>11.0f}  "
                      f"{r['peak_vram_gb']:>6.2f} GB ({vram_pct:.0f}%)")
            except (RuntimeError, Exception):
                print(f"  {bs:>6}  {nw:>7}  OOM")
                torch.cuda.empty_cache()   # release memory so next config can run

    if full_results:
        full_results.sort(key=lambda x: x[2], reverse=True)
        best = full_results[0]
        print()
        print("─" * 70)
        print(f"  RECOMMENDATION")
        print("─" * 70)
        print(f"  BATCH_SIZE  = {best[0]}")
        print(f"  NUM_WORKERS = {best[1]}")
        print(f"  Throughput  = {best[2]:.0f} samples/sec")
        print(f"  Peak VRAM   = {best[3]:.2f} GB / {total_vram:.1f} GB")
        print()
        print("  Paste these values into train_model.py and restart training.")
    print("=" * 70)


if __name__ == "__main__":
    main()

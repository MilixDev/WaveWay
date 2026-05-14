"""Activity classifier training: CSI windows → 5-way softmax.

Uses PyTorch when available, falls back to a pure-numpy SGD path so the
app can train without a torch install.

Inputs (per session, written by DataCollector):
  <dir>/<session>_csi.npy     float32 (N, num_sc)
  <dir>/<session>_labels.npy  int64   (N,)

Sessions without a *_labels.npy file are skipped silently — that's how
legacy pose-format sessions get ignored.

Output:
  <output_path>.npz with w1..b3, num_sc, num_classes, mu, sigma, class_names
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from .activity_classes import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

WINDOW_SIZE: int = 32        # 32 frames @ 10 Hz = 3.2 s of CSI
HIDDEN1:     int = 256
HIDDEN2:     int = 128

EPOCHS:       int = 60
BATCH_SIZE:   int = 64
LR:           float = 1e-3
LR_DECAY:     float = 0.95
VAL_FRACTION: float = 0.20   # held out from the END of each session (time split)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

def _load_dataset(data_dir: Path) -> dict:
    """Read labelled sessions and build train/val arrays.

    Subcarrier count is unified to the mode across sessions (sessions that
    disagree get truncated or zero-padded). z-score statistics come from
    train only — never from val — and are saved in the weights file so
    inference applies the identical transform.
    """
    csi_files = sorted(data_dir.glob("*_csi.npy"))
    if not csi_files:
        raise FileNotFoundError(f"No *_csi.npy files found in {data_dir}")

    sessions = []
    skipped_no_labels = []
    for cf in csi_files:
        name = cf.stem.replace("_csi", "")
        lf = data_dir / f"{name}_labels.npy"
        if not lf.exists():
            skipped_no_labels.append(name)
            continue
        csi    = np.load(str(cf)).astype(np.float32)
        labels = np.load(str(lf)).astype(np.int64)
        if csi.shape[0] != labels.shape[0]:
            logger.warning("Skipping %s — csi/labels mismatch (%d vs %d)",
                           name, csi.shape[0], labels.shape[0])
            continue
        if csi.shape[0] < WINDOW_SIZE + 4:
            logger.warning("Skipping %s — too few samples (%d)", name, csi.shape[0])
            continue
        sessions.append({"name": name, "csi": csi, "labels": labels})

    if skipped_no_labels:
        logger.info("Skipped %d unlabelled session(s): %s",
                    len(skipped_no_labels),
                    ", ".join(skipped_no_labels[:5]) +
                    (" …" if len(skipped_no_labels) > 5 else ""))

    if not sessions:
        raise FileNotFoundError(
            "No labelled sessions. Record sessions with the activity dropdown first."
        )

    sc_counts = [s["csi"].shape[1] for s in sessions]
    num_sc = int(np.bincount(sc_counts).argmax())
    for s in sessions:
        if s["csi"].shape[1] != num_sc:
            logger.warning("Session %s has %d SCs, expected %d — adjusting",
                           s["name"], s["csi"].shape[1], num_sc)
            csi = s["csi"]
            if csi.shape[1] > num_sc:
                s["csi"] = csi[:, :num_sc]
            else:
                pad = np.zeros((csi.shape[0], num_sc - csi.shape[1]), dtype=np.float32)
                s["csi"] = np.concatenate([csi, pad], axis=1)

    # Time-based train/val split inside each session. Sessions too short
    # to split go entirely to train.
    train_s, val_s = [], []
    for s in sessions:
        n = s["csi"].shape[0]
        cut = int(n * (1.0 - VAL_FRACTION))
        if cut < WINDOW_SIZE or (n - cut) < WINDOW_SIZE:
            train_s.append(s)
            continue
        train_s.append({"name": s["name"] + "_t",
                        "csi":  s["csi"][:cut],
                        "labels": s["labels"][:cut]})
        val_s.append({"name": s["name"] + "_v",
                      "csi":  s["csi"][cut:],
                      "labels": s["labels"][cut:]})

    train_csi = np.concatenate([s["csi"] for s in train_s], axis=0)
    mu = train_csi.mean(axis=0).astype(np.float32)
    sigma = train_csi.std(axis=0).astype(np.float32) + 1e-6

    all_train_labels = np.concatenate([s["labels"] for s in train_s], axis=0)
    counts = np.bincount(all_train_labels, minlength=NUM_CLASSES)
    logger.info(
        "Train samples per class: %s",
        ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(NUM_CLASSES)),
    )

    def _build(sess_list) -> Tuple[np.ndarray, np.ndarray]:
        Xs, Ys = [], []
        for s in sess_list:
            csi_z = (s["csi"] - mu) / sigma
            labels = s["labels"]
            N = csi_z.shape[0]
            for i in range(WINDOW_SIZE, N + 1):
                Xs.append(csi_z[i - WINDOW_SIZE: i].flatten())
                Ys.append(int(labels[i - 1]))
        if not Xs:
            return (np.zeros((0, WINDOW_SIZE * num_sc), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        return np.stack(Xs), np.array(Ys, dtype=np.int64)

    X_train, Y_train = _build(train_s)
    X_val,   Y_val   = _build(val_s)

    logger.info("Dataset: num_sc=%d, classes=%d, windows train=%d val=%d",
                num_sc, NUM_CLASSES, len(X_train), len(X_val))

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "num_sc": num_sc, "num_classes": NUM_CLASSES,
        "mu": mu, "sigma": sigma,
    }


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over axis=1."""
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    if logits.shape[0] == 0:
        return 0.0
    return float((logits.argmax(axis=1) == y).mean())


def _shuffle_batches(
    X: np.ndarray, Y: np.ndarray, batch_size: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    idx = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start: start + batch_size]
        yield X[b], Y[b]


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def train(
    data_dir: Path,
    output_path: Path,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> bool:
    logger.info("Loading dataset from %s", data_dir)
    try:
        ds = _load_dataset(data_dir)
    except Exception as exc:
        logger.error("Dataset load failed: %s", exc)
        return False

    if len(ds["X_train"]) < BATCH_SIZE:
        logger.error("Not enough training windows (%d < batch %d)",
                     len(ds["X_train"]), BATCH_SIZE)
        return False

    # Refuse to train on a single class — softmax + cross-entropy degenerate.
    if len(np.unique(ds["Y_train"])) < 2:
        logger.error("Need at least 2 distinct classes in train — record more sessions.")
        return False

    try:
        return _train_torch(ds, output_path, progress_cb)
    except ImportError:
        logger.info("PyTorch not available — using numpy SGD fallback")
        return _train_numpy(ds, output_path, progress_cb)
    except Exception as exc:
        logger.warning("PyTorch training failed (%s) — falling back to numpy", exc)
        return _train_numpy(ds, output_path, progress_cb)


# -----------------------------------------------------------------------
# PyTorch backend
# -----------------------------------------------------------------------

def _train_torch(ds: dict, output_path: Path, progress_cb) -> bool:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    num_sc = ds["num_sc"]
    num_classes = ds["num_classes"]
    input_dim = WINDOW_SIZE * num_sc

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, HIDDEN1), nn.ReLU(),
                nn.Linear(HIDDEN1, HIDDEN2),  nn.ReLU(),
                nn.Linear(HIDDEN2, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("PyTorch training on %s", device)

    # Inverse-frequency class weights so under-represented classes are not
    # drowned out by "vacío" (which dominates if you record longer empty
    # sessions than active ones).
    counts = np.bincount(ds["Y_train"], minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weight_t = torch.tensor(weights, device=device, dtype=torch.float32)

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

    Xt = torch.from_numpy(ds["X_train"]).to(device)
    Yt = torch.from_numpy(ds["Y_train"]).to(device)
    has_val = len(ds["X_val"]) > 0
    if has_val:
        Xv = torch.from_numpy(ds["X_val"]).to(device)
        Yv = torch.from_numpy(ds["Y_val"]).to(device)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        net.train()
        idx = torch.randperm(len(Xt), device=device)
        total, nb = 0.0, 0
        for start in range(0, len(Xt), BATCH_SIZE):
            b = idx[start: start + BATCH_SIZE]
            xb, yb = Xt[b], Yt[b]
            loss = F.cross_entropy(net(xb), yb, weight=weight_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        train_loss = total / max(nb, 1)

        if has_val:
            net.eval()
            with torch.no_grad():
                vp = net(Xv)
                val_loss = float(F.cross_entropy(vp, Yv, weight=weight_t).item())
                val_acc  = float((vp.argmax(dim=1) == Yv).float().mean().item())
        else:
            val_loss, val_acc = train_loss, float("nan")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        logger.info("Epoch %d/%d  train=%.4f  val=%.4f  val_acc=%.3f%s",
                    epoch, EPOCHS, train_loss, val_loss, val_acc,
                    "  *best*" if val_loss == best_val else "")
        if progress_cb:
            progress_cb(epoch, EPOCHS, val_loss)

    sd = {k: v.numpy() for k, v in best_state.items()}
    _save_weights(
        w1=sd["net.0.weight"], b1=sd["net.0.bias"],
        w2=sd["net.2.weight"], b2=sd["net.2.bias"],
        w3=sd["net.4.weight"], b3=sd["net.4.bias"],
        num_sc=num_sc, num_classes=num_classes,
        mu=ds["mu"], sigma=ds["sigma"],
        output_path=output_path,
    )
    return True


# -----------------------------------------------------------------------
# NumPy SGD backend (dependency-free)
# -----------------------------------------------------------------------

def _train_numpy(ds: dict, output_path: Path, progress_cb) -> bool:
    num_sc = ds["num_sc"]
    num_classes = ds["num_classes"]
    input_dim = WINDOW_SIZE * num_sc
    rng = np.random.default_rng(42)

    # He-init for ReLU layers
    s1 = np.sqrt(2.0 / input_dim)
    s2 = np.sqrt(2.0 / HIDDEN1)
    s3 = np.sqrt(2.0 / HIDDEN2)

    w1 = rng.normal(0, s1, (HIDDEN1,    input_dim)).astype(np.float32)
    b1 = np.zeros(HIDDEN1, dtype=np.float32)
    w2 = rng.normal(0, s2, (HIDDEN2,    HIDDEN1)).astype(np.float32)
    b2 = np.zeros(HIDDEN2, dtype=np.float32)
    w3 = rng.normal(0, s3, (num_classes, HIDDEN2)).astype(np.float32)
    b3 = np.zeros(num_classes, dtype=np.float32)

    counts = np.bincount(ds["Y_train"], minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    cls_weights = (counts.sum() / (num_classes * counts)).astype(np.float32)

    X, Y = ds["X_train"], ds["Y_train"]
    has_val = len(ds["X_val"]) > 0
    Xv, Yv = ds["X_val"], ds["Y_val"]

    lr = LR
    best_val = float("inf")
    best = (w1.copy(), b1.copy(), w2.copy(), b2.copy(), w3.copy(), b3.copy())

    for epoch in range(1, EPOCHS + 1):
        total_loss, nb = 0.0, 0
        for xb, yb in _shuffle_batches(X, Y, BATCH_SIZE):
            bs = len(xb)
            h1_pre = xb @ w1.T + b1; h1 = _relu(h1_pre)
            h2_pre = h1 @ w2.T + b2; h2 = _relu(h2_pre)
            logits = h2 @ w3.T + b3

            log_p = _log_softmax(logits)
            sample_w = cls_weights[yb]
            denom = sample_w.sum() + 1e-9
            loss = float((-log_p[np.arange(bs), yb] * sample_w).sum() / denom)
            total_loss += loss
            nb += 1

            # ∂CE/∂logits = (p - onehot), then weighted and normalised
            p = np.exp(log_p)
            d_out = p.copy()
            d_out[np.arange(bs), yb] -= 1.0
            d_out *= sample_w[:, None] / denom

            gw3 = d_out.T @ h2
            gb3 = d_out.sum(axis=0)

            d_h2 = d_out @ w3
            d_h2 *= (h2_pre > 0).astype(np.float32)
            gw2 = d_h2.T @ h1
            gb2 = d_h2.sum(axis=0)

            d_h1 = d_h2 @ w2
            d_h1 *= (h1_pre > 0).astype(np.float32)
            gw1 = d_h1.T @ xb
            gb1 = d_h1.sum(axis=0)

            w1 -= lr * gw1; b1 -= lr * gb1
            w2 -= lr * gw2; b2 -= lr * gb2
            w3 -= lr * gw3; b3 -= lr * gb3

        lr *= LR_DECAY
        train_loss = total_loss / max(nb, 1)

        if has_val:
            h1v = _relu(Xv @ w1.T + b1)
            h2v = _relu(h1v @ w2.T + b2)
            logitsv = h2v @ w3.T + b3
            log_pv = _log_softmax(logitsv)
            sample_wv = cls_weights[Yv]
            denom = sample_wv.sum() + 1e-9
            val_loss = float(
                (-log_pv[np.arange(len(Yv)), Yv] * sample_wv).sum() / denom
            )
            val_acc = _accuracy(logitsv, Yv)
        else:
            val_loss, val_acc = train_loss, float("nan")

        if val_loss < best_val:
            best_val = val_loss
            best = (w1.copy(), b1.copy(), w2.copy(), b2.copy(), w3.copy(), b3.copy())

        logger.info("Epoch %d/%d  train=%.4f  val=%.4f  val_acc=%.3f%s",
                    epoch, EPOCHS, train_loss, val_loss, val_acc,
                    "  *best*" if val_loss == best_val else "")
        if progress_cb:
            progress_cb(epoch, EPOCHS, val_loss)

    w1, b1, w2, b2, w3, b3 = best
    _save_weights(w1, b1, w2, b2, w3, b3, num_sc, num_classes,
                  ds["mu"], ds["sigma"], output_path)
    return True


# -----------------------------------------------------------------------

def _save_weights(
    w1, b1, w2, b2, w3, b3,
    num_sc: int, num_classes: int,
    mu: np.ndarray, sigma: np.ndarray, output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3,
        num_sc=np.array(num_sc),
        num_classes=np.array(num_classes),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        class_names=np.array(CLASS_NAMES),
    )
    logger.info("Weights saved → %s", output_path)

"""Shared data loading, fold assignment and pixel metrics.

The masonry dataset is the public subset released by Dais et al. (2021):
https://github.com/dimitrisdais/crack_detection_CNN_masonry
Set CRACK_DATA to its `dataset` folder, or clone the repository into ./data (see README).
"""
import glob, os, re
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold, KFold

ROOT = os.environ.get("CRACK_DATA", "data/crack_detection_CNN_masonry/dataset")
SEED = 42


def load():
    """Return images X (n,224,224,3), masks Y (n,224,224), source-photograph ids G and file names N."""
    imgs = sorted(glob.glob(f"{ROOT}/crack_detection_224_images/*.png"))
    if not imgs:
        raise FileNotFoundError(f"No images under {ROOT}/crack_detection_224_images; set CRACK_DATA.")
    X, Y, G, N = [], [], [], []
    for f in imgs:
        name = os.path.basename(f)
        X.append(np.array(Image.open(f).convert("RGB")))
        Y.append((np.array(Image.open(f"{ROOT}/crack_detection_224_masks/{name}").convert("L")) > 127).astype(np.uint8))
        G.append(re.sub(r"_\d+\.png$", "", name))  # source photograph id, from the file name
        N.append(name)
    return np.stack(X), np.stack(Y), np.array(G), N


def folds(G, k=5):
    """Main protocol: patches from one source photograph never span training and test."""
    return list(GroupKFold(n_splits=k).split(np.zeros(len(G)), groups=G))


def random_folds(n, k=5):
    """Comparison protocol: patch-level random split that ignores the source photograph (quantifies leakage)."""
    return list(KFold(n_splits=k, shuffle=True, random_state=SEED).split(np.zeros(n)))


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()), tn=int((~pred & ~gt).sum()))


def add(a, b):
    """Sum two confusion counts (pooling over patches)."""
    return {k: a.get(k, 0) + v for k, v in b.items()}


def scores(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(precision=p, recall=r, f1=f1, iou=iou)

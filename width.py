"""Crack-width agreement between predicted and ground-truth masks on held-out photographs.

Width at a skeleton pixel = 2 x Euclidean distance to the background (in pixels).
Per patch we compare the median and the 90th-percentile width of the predicted crack
with those of the expert mask. No physical scale is available in this dataset, so all
widths are in pixels; with a known ground sampling distance they convert directly to mm.
"""
import sys, json, glob
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
sys.path.insert(0, ".")
from common import *
import classical_seg as cl

def widths(mask):
    m = mask.astype(bool)
    if m.sum() < 10:
        return None
    dist = ndimage.distance_transform_edt(m)
    sk = skeletonize(m)
    w = 2 * dist[sk]
    return w[w > 0] if (w > 0).any() else None

def stats(w):
    return (float(np.median(w)), float(np.percentile(w, 90)))

X, Y, G, N = load()
F = folds(G)
fold_of = {j: i for i, (_, te) in enumerate(F) for j in te}
cr = json.load(open("results_classical.json"))

methods = {"Black-hat + threshold": {N[j]: cl.segment(X[j], *cr["per_fold"][fold_of[j]]["params"]) for j in range(len(N))}}
names = {"Unet_resnet18": "U-Net (ResNet-18, from scratch)", "DeepLabV3Plus_resnet18": "DeepLabV3+ (ResNet-18, from scratch)",
         "Unet_resnet18_imagenet": "U-Net (ResNet-18, ImageNet-pretrained)", "DeepLabV3Plus_resnet18_imagenet": "DeepLabV3+ (ResNet-18, ImageNet-pretrained)"}
for f in sorted(glob.glob("preds_*.npz")):
    k = f[6:-4]
    if k not in names:  # only the four main models (grouped split, seed 42)
        continue
    P = np.load(f)
    methods[names.get(k, k)] = {n: P[n].astype(bool) for n in P.files}

gt = {N[j]: widths(Y[j]) for j in range(len(N))}
gt_ok = [n for n in N if gt[n] is not None]
out = {"n_patches_with_gt_crack": len(gt_ok), "gt_median_width_px_median": float(np.median([stats(gt[n])[0] for n in gt_ok])), "methods": {}}
for name, pr in methods.items():
    e_med, e_p90, miss = [], [], 0
    for n in gt_ok:
        w = widths(pr[n])
        if w is None:
            miss += 1
            continue
        g, p = stats(gt[n]), stats(w)
        e_med.append(p[0] - g[0]); e_p90.append(p[1] - g[1])
    e_med, e_p90 = np.array(e_med), np.array(e_p90)
    out["methods"][name] = dict(
        patches_compared=int(len(e_med)), patches_no_detection=miss,
        median_width_MAE_px=float(np.abs(e_med).mean()), median_width_bias_px=float(e_med.mean()),
        p90_width_MAE_px=float(np.abs(e_p90).mean()), p90_width_bias_px=float(e_p90.mean()))
json.dump(out, open("results_width.json", "w"), indent=1)
print(json.dumps(out, indent=1))

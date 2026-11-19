"""Manuscript figures (Figures 1-8) from the stored results (no retraining). Output: figures/*.png at 300 dpi.

Usage: python3 make_figures.py /path/to/DeepCrack   (folder containing test_img/ and test_lab/)
"""
import sys, json, os, glob
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
import segmentation_models_pytorch as smp
sys.path.insert(0, ".")
from common import *

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "figure.facecolor": "white", "hatch.linewidth": 0.6})
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b"
FOLDC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# method: (label, file tag, colour, pretrained)
M = [("U-Net, pretrained", "Unet_resnet18_imagenet", BLUE, True), ("U-Net, from scratch", "Unet_resnet18", BLUE, False),
     ("DeepLabV3+, pretrained", "DeepLabV3Plus_resnet18_imagenet", ORANGE, True), ("DeepLabV3+, from scratch", "DeepLabV3Plus_resnet18", ORANGE, False)]
BASE = "Black-hat + threshold"
def style(c, pre):  # pretrained = solid; from scratch = white fill with hatching in the same hue
    return dict(facecolor=c, edgecolor=c) if pre else dict(facecolor="white", edgecolor=c, hatch="////")
def res(tag): return json.load(open(f"results_{tag}.json"))
def panel(ax, s): ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
def legend_methods(ax, base=True, **kw):
    h = [Patch(**style(c, p), label=l) for l, _, c, p in M] + ([Patch(facecolor=GREY, edgecolor=GREY, label=BASE)] if base else [])
    ax.legend(handles=h, frameon=False, fontsize=8, **kw)

X, Y, G, N = load()
F = folds(G); RF = random_folds(len(G))

# ---------- Figure 1: workflow ----------
fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
def box(x, y, w, h, t, fc="#f3f2ee", ec="#52514e", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=INK)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9))
box(1, 29, 19, 15, "Masonry dataset [4]\n240 patches, 224 × 224 px\n76 photographs\nexpert crack masks")
box(1, 5, 19, 13, "DeepCrack test set [7]\n237 images, 544 × 384 px\nused for testing only")
box(27, 29, 19, 15, "Five-fold split\ngrouped by photograph\n(main protocol)\nrandom by patch\n(comparison)")
box(53, 34, 19, 10, "Morphological baseline\nblack-hat + threshold")
box(53, 20, 19, 12, "U-Net and DeepLabV3+\nResNet-18 encoder\nfrom scratch / pretrained\nthree training seeds")
box(79, 38, 20, 7.5, "Segmentation on unseen\nphotographs (Section 4.1)", fc="#e8f0fb", ec=BLUE)
box(79, 28.5, 20, 7.5, "Crack-width agreement\n(Section 4.2)", fc="#e8f0fb", ec=BLUE)
box(79, 19, 20, 7.5, "Effect of the data split\n(Section 4.5)", fc="#e8f0fb", ec=BLUE)
box(79, 7, 20, 9, "Cross-dataset test\nwithout retraining\n(Section 4.3)", fc="#e8f0fb", ec=BLUE)
arrow(20.5, 36.5, 26.5, 36.5); arrow(46.5, 38.5, 52.5, 39); arrow(46.5, 33, 52.5, 27)
for yy in (41.7, 32.2, 22.7): arrow(72.5, 30, 78.5, yy)
arrow(72.5, 39, 78.5, 41.7); arrow(20.5, 11.5, 78.5, 11.5); arrow(62.5, 19.5, 62.5, 12.2)
ax.text(64, 15.5, "five-fold ensemble", fontsize=7, color="#52514e", va="center")
for x, t in ((10.5, "Data"), (36.5, "Protocol"), (62.5, "Methods"), (89, "Evaluation")):
    ax.text(x, 48.5, t, ha="center", fontsize=9, fontweight="bold", color=INK)
fig.savefig("figures/Figure_1_workflow.png"); plt.close(fig)

# ---------- Figure 2: data examples ----------
DC = sys.argv[1] if len(sys.argv) > 1 else "data/DeepCrack"
dimgs = sorted(glob.glob(f"{DC}/test_img/*.jpg"))
def overlay(img, m):
    o = img.astype(float).copy(); o[m > 0] = 0.45 * o[m > 0] + 0.55 * np.array([255, 40, 40]); return o.astype(np.uint8)
frac = Y.reshape(len(Y), -1).mean(1)
pick = [int(np.argsort(frac)[int(q * (len(frac) - 1))]) for q in (0.1, 0.4, 0.7, 0.95)]
fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.28, wspace=0.06)
for i, j in enumerate(pick):
    a = fig.add_subplot(gs[0, i]) if i < 4 else None
    a.imshow(X[j]); a.axis("off"); a.set_title(f"{100*frac[j]:.1f}% crack", fontsize=7.5, pad=2)
    b = fig.add_subplot(gs[1, i]); b.imshow(overlay(X[j], Y[j])); b.axis("off")
    if i == 0: panel(a, "(a)")
for k, idx in enumerate((10, 120)):
    im = np.array(Image.open(dimgs[idx]).convert("RGB")); lb = (np.array(Image.open(f"{DC}/test_lab/{os.path.basename(dimgs[idx])[:-4]}.png").convert("L")) > 127)
    a = fig.add_subplot(gs[0, 4 + k]); a.imshow(im); a.axis("off"); a.set_aspect("equal")
    b = fig.add_subplot(gs[1, 4 + k]); b.imshow(overlay(im, lb)); b.axis("off")
    if k == 0: panel(a, "(b)")
fig.savefig("figures/Figure_2_datasets.png"); plt.close(fig)

# ---------- Figure 3: protocol (fold assignment per photograph) ----------
ug, cnt = np.unique(G, return_counts=True); order = ug[np.argsort(-cnt, kind="stable")]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
for ax, FF, t, tag in ((axs[0], F, "Grouped by source photograph (main protocol)", "(a)"), (axs[1], RF, "Random by patch (comparison)", "(b)")):
    fold_of = np.empty(len(G), int)
    for i, (_, te) in enumerate(FF): fold_of[te] = i
    bottom = np.zeros(len(order))
    for f in range(5):
        h = np.array([((G == g) & (fold_of == f)).sum() for g in order])
        ax.bar(np.arange(len(order)), h, bottom=bottom, color=FOLDC[f], width=0.82, edgecolor="white", linewidth=0.4, label=f"Fold {f+1}"); bottom += h
    ax.set_ylabel("Patches"); ax.set_title(t, fontsize=9, loc="left"); ax.grid(axis="x", visible=False); panel(ax, tag)
axs[0].legend(frameon=False, ncol=5, fontsize=8, loc="upper right", title="Test fold of each patch", title_fontsize=8)
axs[1].set_xlabel("Source photographs, ordered by number of patches (76 photographs)"); axs[1].set_xticks([])
fig.tight_layout(); fig.savefig("figures/Figure_3_protocol.png"); plt.close(fig)

# ---------- Figure 4: segmentation metrics + per-fold F1 (first run, grouped) ----------
cl = res("classical")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axs[0]; mets = ["precision", "recall", "f1", "iou"]; w = 0.16
entries = [(BASE, cl, dict(facecolor=GREY, edgecolor=GREY))] + [(l, res(t), style(c, p)) for l, t, c, p in reversed(M)]
for k, (l, d, st) in enumerate(entries):
    v = [d["pooled"][m] for m in mets]
    ax.bar(np.arange(4) + (k - 2) * w, v, w * 0.9, linewidth=0.8, **st)
    for xx, vv in zip(np.arange(4) + (k - 2) * w, v):
        if k in (0, 4): ax.text(xx, vv + 0.012, f"{vv:.2f}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(["Precision", "Recall", "F1", "IoU"]); ax.set_ylim(0, 1); ax.set_ylabel("Pooled score"); ax.grid(axis="x", visible=False)
legend_methods(ax, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=1, handlelength=1.4); panel(ax, "(a)")
ax = axs[1]
rows = [(BASE, cl, GREY, True)] + [(l, res(t), c, p) for l, t, c, p in reversed(M)]
for k, (l, d, c, p) in enumerate(rows):
    f1 = [x["f1"] for x in d["per_fold"]]
    ax.scatter(f1, [k] * 5, s=26, facecolor=c if p else "white", edgecolor=c, linewidth=1, zorder=3)
    ax.plot([np.mean(f1)] * 2, [k - 0.28, k + 0.28], color=INK, lw=1.2, zorder=4)
ax.set_yticks(range(5)); ax.set_yticklabels(["Baseline", "DeepLabV3+\nscratch", "DeepLabV3+\npretrained", "U-Net\nscratch", "U-Net\npretrained"], fontsize=7.5)
ax.set_xlabel("F1 per test fold (bar: mean)"); ax.set_xlim(0.4, 0.9); ax.grid(axis="y", visible=False); panel(ax, "(b)")
fig.tight_layout(); fig.savefig("figures/Figure_4_segmentation.png"); plt.close(fig)

# ---------- Figure 6: crack width ----------
def widths(mask):
    m = mask.astype(bool)
    if m.sum() < 10: return None
    w = 2 * ndimage.distance_transform_edt(m)[skeletonize(m)]
    return w[w > 0] if (w > 0).any() else None
gt = {N[j]: widths(Y[j]) for j in range(len(N))}
import classical_seg as cs
fold_of = {j: i for i, (_, te) in enumerate(F) for j in te}
preds = {BASE: {N[j]: cs.segment(X[j], *cl["per_fold"][fold_of[j]]["params"]) for j in range(len(N))}}
for l, t, c, p in M: preds[l] = dict(np.load(f"preds_{t}.npz"))
err, pairs = {}, {}
for l, pr in preds.items():
    e, pp = [], []
    for n in N:
        w = widths(pr[n])
        if gt[n] is None or w is None: continue
        e.append(np.median(w) - np.median(gt[n])); pp.append((np.median(gt[n]), np.median(w)))
    err[l], pairs[l] = np.array(e), np.array(pp)
    print("width", l, len(e), "MAE %.2f bias %.2f" % (np.abs(e).mean(), np.mean(e)))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]; pp = pairs["U-Net, pretrained"]; lim = float(np.ceil(max(pp.max(), 1) / 5) * 5)
ax.plot([0, lim], [0, lim], color="#52514e", lw=0.9, ls="--", zorder=2); ax.text(lim * 0.97, lim * 0.9, "1:1", fontsize=7.5, color="#52514e", ha="right")
ax.scatter(pp[:, 0], pp[:, 1], s=14, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.4, zorder=3)
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.set_xlabel("Median width, expert mask (pixels)"); ax.set_ylabel("Median width, predicted mask (pixels)")
ax.set_title(f"U-Net, pretrained (n = {len(pp)} patches)", fontsize=8.5, loc="left"); panel(ax, "(a)")
ax = axs[1]; order_w = [BASE] + [l for l, *_ in reversed(M)]
bp = ax.boxplot([err[l] for l in order_w], vert=False, widths=0.55, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#b5b4ae", markeredgecolor="none"), medianprops=dict(color=INK, lw=1.2),
                whiskerprops=dict(color="#52514e", lw=0.8), capprops=dict(color="#52514e", lw=0.8))
sty = [dict(facecolor=GREY, edgecolor=GREY)] + [style(c, p) for _, _, c, p in reversed(M)]
for b, s in zip(bp["boxes"], sty): b.set(linewidth=0.9, **s); b.set_alpha(0.9)
ax.axvline(0, color="#52514e", lw=0.9, ls="--")
ax.set_yticks(range(1, 6)); ax.set_yticklabels(["Baseline", "DeepLabV3+\nscratch", "DeepLabV3+\npretrained", "U-Net\nscratch", "U-Net\npretrained"], fontsize=7.5)
ax.set_xlabel("Error in median width, predicted − expert (pixels)"); ax.grid(axis="y", visible=False); panel(ax, "(b)")
fig.tight_layout(); fig.savefig("figures/Figure_6_width.png"); plt.close(fig)

# ---------- Figure 5: example predictions on held-out photographs ----------
cols = [("Image", None), ("Expert mask", None), ("Baseline", BASE)] + [(l.replace(", ", "\n"), l) for l, *_ in M]
ex_idx = [int(np.argsort(frac)[int(q * (len(frac) - 1))]) for q in (0.2, 0.5, 0.8, 0.97)]
fig, axs = plt.subplots(len(ex_idx), len(cols), figsize=(7.2, 1.15 * len(ex_idx) + 0.4))
for r, j in enumerate(ex_idx):
    for c_, (t, key) in enumerate(cols):
        a = axs[r, c_]
        img = X[j] if c_ == 0 else (Y[j] if c_ == 1 else preds[key][N[j]])
        a.imshow(img, cmap=None if c_ == 0 else "gray"); a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(t, fontsize=7.5, pad=3)
fig.subplots_adjust(wspace=0.04, hspace=0.06)
fig.savefig("figures/Figure_5_qualitative.png"); plt.close(fig)

# ---------- Figure 7: cross-dataset ----------
cd = json.load(open("results_crossdomain.json"))["results"]
key = {"U-Net, pretrained": "U-Net (ResNet-18, ImageNet-pretrained)", "U-Net, from scratch": "U-Net (ResNet-18, from scratch)",
       "DeepLabV3+, pretrained": "DeepLabV3+ (ResNet-18, ImageNet-pretrained)", "DeepLabV3+, from scratch": "DeepLabV3+ (ResNet-18, from scratch)"}
models = []
for pth in sorted(glob.glob("model_Unet_resnet18_imagenet_fold*.pt")):
    m = smp.Unet("resnet18", encoder_weights=None, classes=1); m.load_state_dict(torch.load(pth, map_location="cpu")); m.eval(); models.append(m)
mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.set_num_threads(2); ex = []
for idx in (10, 120, 200):
    im = np.array(Image.open(dimgs[idx]).convert("RGB")); lb = (np.array(Image.open(f"{DC}/test_lab/{os.path.basename(dimgs[idx])[:-4]}.png").convert("L")) > 127)
    with torch.no_grad():
        xt = (torch.tensor(im).permute(2, 0, 1)[None].float() / 255.0 - mean) / std
        pr = torch.stack([torch.sigmoid(m(xt))[0, 0] for m in models]).mean(0).numpy() > 0.5
    ex.append((im, lb, pr)); print("deepcrack example", idx, scores(confusion(pr, lb)))
fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(4, 3, height_ratios=[1.55, 1, 1, 1], hspace=0.12, wspace=0.05)
ax = fig.add_subplot(gs[0, :]); mets = ["precision", "recall", "f1", "iou"]
entries = [(BASE, cd[BASE], dict(facecolor=GREY, edgecolor=GREY))] + [(l, cd[key[l]], style(c, p)) for l, t, c, p in reversed(M)]
for k, (l, d, st) in enumerate(entries):
    v = [d[m] for m in mets]; ax.bar(np.arange(4) + (k - 2) * w, v, w * 0.9, linewidth=0.8, **st)
    for xx, vv in zip(np.arange(4) + (k - 2) * w, v):
        if k in (0, 4): ax.text(xx, vv + 0.012, f"{vv:.2f}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(["Precision", "Recall", "F1", "IoU"]); ax.set_ylim(0, 1.05); ax.set_ylabel("Pooled score"); ax.grid(axis="x", visible=False)
legend_methods(ax, loc="center left", bbox_to_anchor=(1.0, 0.5)); panel(ax, "(a)")
for r, (im, lb, pr) in enumerate(ex):
    for c_, (img, t) in enumerate(((im, "Image"), (lb, "Annotation"), (pr, "U-Net, pretrained (masonry-trained)"))):
        a = fig.add_subplot(gs[r + 1, c_]); a.imshow(img, cmap="gray"); a.axis("off")
        if r == 0: a.set_title(t, fontsize=8, pad=3)
        if r == 0 and c_ == 0: panel(a, "(b)")
fig.savefig("figures/Figure_7_crossdataset.png"); plt.close(fig)

# ---------- Figure 8: grouped vs random split, three seeds ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axs[0]; ax2 = axs[1]; pos = []
for k, (l, t, c, p) in enumerate(reversed(M)):
    g = [res(t + s)["pooled"]["f1"] for s in ("", "_seed1", "_seed2")]; r = [res(t + "_randsplit" + s)["pooled"]["f1"] for s in ("", "_seed1", "_seed2")]
    ax.plot([np.mean(g), np.mean(r)], [k, k], color=c, lw=1.6, zorder=2)
    ax.scatter(g, [k] * 3, s=30, marker="o", facecolor="white", edgecolor=c, linewidth=1.1, zorder=3)
    ax.scatter(r, [k] * 3, s=30, marker="s", facecolor=c, edgecolor=c, linewidth=1.1, zorder=3)
    ax.text(np.mean(r) + 0.006, k + 0.22, f"+{np.mean(r) - np.mean(g):.3f}", fontsize=7.5, color=INK, va="center")
    fg = np.concatenate([[x["f1"] for x in res(t + s)["per_fold"]] for s in ("", "_seed1", "_seed2")])
    fr = np.concatenate([[x["f1"] for x in res(t + "_randsplit" + s)["per_fold"]] for s in ("", "_seed1", "_seed2")])
    b = ax2.boxplot([fg, fr], positions=[k + 0.19, k - 0.19], vert=False, widths=0.3, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#b5b4ae", markeredgecolor="none"), medianprops=dict(color=INK, lw=1.1),
                    whiskerprops=dict(color="#52514e", lw=0.8), capprops=dict(color="#52514e", lw=0.8))
    b["boxes"][0].set(facecolor="white", edgecolor=c, linewidth=1); b["boxes"][1].set(facecolor=c, edgecolor=c, linewidth=1, alpha=0.9)
labs = ["DeepLabV3+\nscratch", "DeepLabV3+\npretrained", "U-Net\nscratch", "U-Net\npretrained"]
ax.set_yticks(range(4)); ax.set_yticklabels(labs, fontsize=7.5); ax.set_ylim(-0.6, 3.75); ax.set_xlim(0.72, 0.84); ax.set_xlabel("Pooled F1 (one marker per training seed)"); ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, label="Grouped by photograph"),
                   Line2D([], [], marker="s", ls="", markerfacecolor=INK, markeredgecolor=INK, label="Random by patch")], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
panel(ax, "(a)")
ax2.set_yticks(range(4)); ax2.set_yticklabels(labs, fontsize=7.5); ax2.set_ylim(-0.6, 3.75); ax2.set_xlabel("F1 per test fold (5 folds × 3 seeds)"); ax2.grid(axis="y", visible=False)
ax2.legend(handles=[Patch(facecolor="white", edgecolor=INK, label="Grouped by photograph"), Patch(facecolor=INK, edgecolor=INK, label="Random by patch")], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
panel(ax2, "(b)")
fig.tight_layout(); fig.savefig("figures/Figure_8_split.png"); plt.close(fig)
print("done")

"""Train a segmentation network on each of the five folds and score it on the held-out patches.

Usage examples:
    python3 train.py --arch Unet                                         # U-Net, from scratch, grouped split
    python3 train.py --arch DeepLabV3Plus --pretrained weights/resnet18_a1_0-d63eafa0.pth
    python3 train.py --arch Unet --split random --seed 1                 # leakage comparison, second seed
Outputs results_<tag>.json (per-fold and pooled metrics) and preds_<tag>.npz (binary masks of the held-out patches);
for the grouped split with the default seed it also saves the five fold models (model_<tag>_fold<i>.pt).
"""
import sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
sys.path.insert(0, ".")
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="Unet", choices=["Unet", "DeepLabV3Plus"])
ap.add_argument("--encoder", default="resnet18")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--pretrained", default="", help="path to ImageNet ResNet-18 weights (timm 'resnet18.a1_in1k'); empty = from scratch")
ap.add_argument("--split", default="grouped", choices=["grouped", "random"],
                help="grouped: by source photograph (main protocol); random: patch-level KFold, for the leakage comparison only")
ap.add_argument("--seed", type=int, default=SEED,
                help="training seed (initialisation, batch order, augmentation); the fold assignment is always fixed")
a = ap.parse_args()
torch.set_num_threads(2); torch.manual_seed(a.seed); np.random.seed(a.seed)

X, Y, G, N = load()
Xt = torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0
if a.pretrained:  # ImageNet statistics to match the pretrained encoder
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
else:             # statistics of the masonry dataset
    mean, std = Xt.mean((0, 2, 3), keepdim=True), Xt.std((0, 2, 3), keepdim=True)
Xt = (Xt - mean) / std
Yt = torch.tensor(Y).unsqueeze(1).float()


def augment(x, y):
    """Random flips, rotations by multiples of 90 degrees, brightness/contrast change of up to +-10%."""
    if np.random.rand() < .5: x, y = x.flip(-1), y.flip(-1)
    if np.random.rand() < .5: x, y = x.flip(-2), y.flip(-2)
    k = np.random.randint(4); x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    x = x * (1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1) - .5)) + 0.2 * (torch.rand(x.shape[0], 1, 1, 1) - .5)
    return x, y


def loss_fn(logit, y):
    """Binary cross-entropy + Dice loss."""
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return F.binary_cross_entropy_with_logits(logit, y) + dice


tag = (f"{a.arch}_{a.encoder}" + ("_imagenet" if a.pretrained else "") + ("_randsplit" if a.split == "random" else "")
       + (f"_seed{a.seed}" if a.seed != SEED else ""))
FOLDS = folds(G) if a.split == "grouped" else random_folds(len(G))
ENC = {k: v for k, v in torch.load(a.pretrained, map_location="cpu").items() if not k.startswith("fc.")} if a.pretrained else None
MAX_LR = 1e-3 if a.pretrained else 3e-3
tot, per_fold, times, preds = {}, [], [], {}
for i, (tr, te) in enumerate(FOLDS):
    model = getattr(smp, a.arch)(a.encoder, encoder_weights=None, classes=1)
    if ENC is not None: model.encoder.load_state_dict(ENC)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, MAX_LR, total_steps=a.epochs * int(np.ceil(len(tr) / a.bs)))
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr); L = 0
        for b in range(0, len(perm), a.bs):
            idx = perm[b:b + a.bs]; x, y = augment(Xt[idx], Yt[idx])
            l = loss_fn(model(x), y); opt.zero_grad(); l.backward(); opt.step(); sched.step(); L += l.item() * len(idx)
        print(f"[{tag}] fold {i} epoch {ep+1}/{a.epochs} loss {L/len(tr):.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval(); c = {}
    with torch.no_grad():  # model after the final epoch, fixed threshold 0.5: no choice depends on test data
        for j in te:
            s = time.perf_counter(); pr = (torch.sigmoid(model(Xt[j:j+1]))[0, 0] > .5).numpy(); times.append(time.perf_counter() - s)
            c = add(c, confusion(pr, Y[j])); preds[N[j]] = pr
    tot = add(tot, c); per_fold.append(dict(fold=i, train_s=time.time() - t0, **scores(c)))
    print(per_fold[-1], flush=True)
    if a.split == "grouped" and a.seed == SEED: torch.save(model.state_dict(), f"model_{tag}_fold{i}.pt")

out = dict(split=a.split, seed=a.seed, model=f"{a.arch} ({a.encoder}, " + ("ImageNet-pretrained" if a.pretrained else "from scratch") + ")",
           epochs=a.epochs, params=sum(p.numel() for p in model.parameters()),
           per_fold=per_fold, pooled=scores(tot), ms_per_patch=1000 * float(np.mean(times)))
json.dump(out, open(f"results_{tag}.json", "w"), indent=1)
np.savez_compressed(f"preds_{tag}.npz", **{k: v.astype(np.uint8) for k, v in preds.items()})
print("POOLED", out["pooled"], out["ms_per_patch"])

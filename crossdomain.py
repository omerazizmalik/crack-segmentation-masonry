"""Cross-dataset test: models trained only on masonry patches, applied unchanged to the DeepCrack test set.

DeepCrack (Liu et al., Neurocomputing 2019): 237 test images, 544x384 px, cracks in concrete,
pavement and other surfaces. No retraining or threshold tuning on DeepCrack. For each learned model
the five fold models are averaged (probability ensemble). Usage:
    python3 crossdomain.py /path/to/deepcrack_x
"""
import sys, json, glob, os, collections, time
import numpy as np, torch
from PIL import Image
import segmentation_models_pytorch as smp
sys.path.insert(0, ".")
from common import *
import classical_seg as cl

root = sys.argv[1]
torch.set_num_threads(2)
imgs = sorted(glob.glob(os.path.join(root, "test_img", "*.jpg")))
labs = [os.path.join(root, "test_lab", os.path.basename(f)[:-4] + ".png") for f in imgs]
X = [np.array(Image.open(f).convert("RGB")) for f in imgs]
Y = [(np.array(Image.open(f).convert("L")) > 127).astype(np.uint8) for f in labs]

Xm, _, _, _ = load()  # masonry training statistics for the from-scratch models
Xm = torch.tensor(Xm).permute(0, 3, 1, 2).float() / 255.0
norm = {False: (Xm.mean((0, 2, 3), keepdim=True), Xm.std((0, 2, 3), keepdim=True)),
        True: (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))}

results = {}
# classical: the parameter set chosen most often across the five masonry folds
cr = json.load(open("results_classical.json"))
p = collections.Counter(tuple(f["params"]) for f in cr["per_fold"]).most_common(1)[0][0]
c = {}
for x, y in zip(X, Y):
    c = add(c, confusion(cl.segment(x, *p), y))
results["Black-hat + threshold"] = scores(c)
print("classical", results["Black-hat + threshold"], flush=True)

specs = [("Unet", False), ("DeepLabV3Plus", False), ("Unet", True), ("DeepLabV3Plus", True)]
label = {("Unet", False): "U-Net (ResNet-18, from scratch)", ("DeepLabV3Plus", False): "DeepLabV3+ (ResNet-18, from scratch)",
         ("Unet", True): "U-Net (ResNet-18, ImageNet-pretrained)", ("DeepLabV3Plus", True): "DeepLabV3+ (ResNet-18, ImageNet-pretrained)"}
for arch, pre in specs:
    tag = f"{arch}_resnet18" + ("_imagenet" if pre else "")
    paths = sorted(glob.glob(f"model_{tag}_fold*.pt"))
    if len(paths) != 5:
        print("skip", tag, len(paths)); continue
    models = []
    for pth in paths:
        m = getattr(smp, arch)("resnet18", encoder_weights=None, classes=1)
        m.load_state_dict(torch.load(pth, map_location="cpu")); m.eval(); models.append(m)
    mean, std = norm[pre]
    c, t0 = {}, time.time()
    with torch.no_grad():
        for x, y in zip(X, Y):
            xt = (torch.tensor(x).permute(2, 0, 1)[None].float() / 255.0 - mean) / std
            prob = torch.stack([torch.sigmoid(m(xt))[0, 0] for m in models]).mean(0).numpy()
            c = add(c, confusion(prob > 0.5, y))
    results[label[(arch, pre)]] = scores(c)
    print(tag, results[label[(arch, pre)]], f"{time.time()-t0:.0f}s", flush=True)

json.dump(dict(dataset="DeepCrack test (237 images)", trained_on="Dais et al. masonry patches only", results=results),
          open("results_crossdomain.json", "w"), indent=1)

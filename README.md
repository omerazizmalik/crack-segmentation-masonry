# Crack segmentation for masonry: evaluation on unseen photographs and across datasets

Code accompanying the paper

> Aziz, O.; Mendes, N. *Deep Learning Crack Segmentation for the Condition Assessment of Masonry Structures: Evaluation on Unseen Photographs and Across Datasets.* Submitted to *Applied Sciences* (MDPI), 2026.

The scripts reproduce every number, table and figure in the paper from public data:

- five-fold cross-validation **grouped by source photograph**, so that test patches always come from photographs not seen in training;
- a **random patch-level split** of the same data, to measure the effect of data leakage;
- a morphological baseline (black-hat + threshold) and four networks (U-Net and DeepLabV3+, ResNet-18 encoder, from scratch and ImageNet-pretrained), each network trained with three seeds;
- crack-width agreement between predicted and expert masks (pixels);
- a cross-dataset test on DeepCrack without retraining.

## Data (not included in this repository)

| Dataset | Source | Use |
|---|---|---|
| Masonry crack patches, 240 patches of 224 × 224 px | Dais et al. (2021), https://github.com/dimitrisdais/crack_detection_CNN_masonry (folder `dataset`) | training and testing |
| DeepCrack, 237 test images | Liu et al. (2019), https://github.com/yhlleo/DeepCrack (folder `dataset`; non-commercial research use only) | cross-dataset test only |

```bash
mkdir -p data && cd data
git clone --depth 1 https://github.com/dimitrisdais/crack_detection_CNN_masonry.git
# DeepCrack: download and unzip the dataset from https://github.com/yhlleo/DeepCrack into data/DeepCrack
# so that data/DeepCrack/test_img and data/DeepCrack/test_lab exist
cd ..
```

The source photograph of each masonry patch is taken from its file name (`<photo>_<n>.png`). The resulting fold assignment for both protocols is given in `folds.csv`.

## Pretrained weights

The pretrained models use the ImageNet ResNet-18 weights of Wightman et al. (timm `resnet18.a1_in1k`):

```bash
mkdir -p weights
wget -O weights/resnet18_a1_0-d63eafa0.pth \
  https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth
```

## Installation

```bash
pip install -r requirements.txt
```

Experiments in the paper were run on a CPU (Intel Xeon 2.1 GHz, two threads) with Segmentation Models PyTorch 0.5.0, scikit-learn 1.9.1, scikit-image 0.26.0 and OpenCV 5.0.0.

## Running

```bash
export CRACK_DATA=data/crack_detection_CNN_masonry/dataset
bash run_all.sh            # everything, about 20 hours on two CPU threads
```

or step by step:

| Script | What it does | Output |
|---|---|---|
| `export_folds.py` | fold assignment of every patch, both protocols | `folds.csv` |
| `classical.py [--split random]` | baseline, parameters tuned on the training folds only | `results_classical*.json` (copies in `results/`) |
| `train.py --arch {Unet,DeepLabV3Plus} [--pretrained W] [--split random] [--seed S]` | trains and tests one network on the five folds | `results_<tag>.json`, `preds_<tag>.npz`, fold models |
| `width.py` | crack-width agreement (median and 90th-percentile width, pixels) | `results_width.json` |
| `crossdomain.py <DeepCrack folder>` | masonry-trained models applied to DeepCrack (five-fold ensemble) | `results_crossdomain.json` |
| `make_figures.py <DeepCrack folder>` | Figures 1–8 of the paper | `figures/*.png` |

Common code (data loading, folds, metrics) is in `common.py`; the baseline itself is in `classical_seg.py`.

All metrics are pixel-level and pooled over the held-out patches of all five folds; per-fold values are also stored. The model after the final epoch and a fixed threshold of 0.5 are used, so no choice depends on the test data.

## Licence

Code: MIT (see `LICENSE`). The datasets are not redistributed here and remain under the terms set by their authors.

## Citation

Please cite the paper above, and the datasets:

- Dais, D.; Bal, İ.E.; Smyrou, E.; Sarhosis, V. Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning. *Autom. Constr.* **2021**, *125*, 103606.
- Liu, Y.; Yao, J.; Lu, X.; Xie, R.; Li, L. DeepCrack: A deep hierarchical feature learning architecture for crack segmentation. *Neurocomputing* **2019**, *338*, 139–153.

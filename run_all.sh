#!/bin/bash
# Reproduces all experiments of the paper. Run from the repository root.
# CPU, two threads: about one hour per network and split; the full set (16 network runs + 4 more for seeds) takes ~20 hours.
set -e
export CRACK_DATA=${CRACK_DATA:-data/crack_detection_CNN_masonry/dataset}
W=weights/resnet18_a1_0-d63eafa0.pth   # ImageNet ResNet-18 (timm resnet18.a1_in1k)
DEEPCRACK=${DEEPCRACK:-data/DeepCrack}  # folder containing test_img/ and test_lab/

python3 export_folds.py

# 1. Morphological baseline, both protocols
python3 classical.py
python3 classical.py --split random

# 2. Networks: two architectures x {from scratch, pretrained} x {grouped, random} x seeds {42, 1, 2}
for seed in 42 1 2; do
  for arch in Unet DeepLabV3Plus; do
    for split in grouped random; do
      python3 train.py --arch $arch --split $split --seed $seed
      python3 train.py --arch $arch --split $split --seed $seed --pretrained $W
    done
  done
done

# 3. Crack-width agreement and cross-dataset test (use the seed-42, grouped-split models)
python3 width.py
python3 crossdomain.py $DEEPCRACK

# 4. Figures
python3 make_figures.py $DEEPCRACK

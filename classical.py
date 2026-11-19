"""Baseline evaluation: parameters chosen per fold by exhaustive search on the training patches only (pooled F1).

Usage:
    python3 classical.py                 # grouped by source photograph -> results_classical.json
    python3 classical.py --split random  # patch-level random split   -> results_classical_randsplit.json
"""
import sys, json, time, itertools, argparse
import numpy as np
sys.path.insert(0, ".")
from common import *
from classical_seg import segment, GRID_K, GRID_T, GRID_A

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="grouped", choices=["grouped", "random"])
a = ap.parse_args()

X, Y, G, N = load()
FOLDS = folds(G) if a.split == "grouped" else random_folds(len(G))
grid = list(itertools.product(GRID_K, GRID_T, GRID_A))
cache = {p: [confusion(segment(X[j], *p), Y[j]) for j in range(len(N))] for p in grid}  # every setting on every patch

tot, per_fold, times = {}, [], []
for i, (tr, te) in enumerate(FOLDS):
    def train_f1(p):
        c = {}
        for j in tr: c = add(c, cache[p][j])
        return scores(c)["f1"]
    best = max(grid, key=train_f1)
    c = {}
    for j in te:
        s = time.perf_counter(); pr = segment(X[j], *best); times.append(time.perf_counter() - s)
        c = add(c, confusion(pr, Y[j]))
    tot = add(tot, c)
    per_fold.append(dict(fold=i, params=list(best), **scores(c)))
    print(per_fold[-1], flush=True)

out = dict(split=a.split, model="Black-hat + threshold", per_fold=per_fold, pooled=scores(tot), ms_per_patch=1000 * float(np.mean(times)))
json.dump(out, open("results_classical" + ("_randsplit" if a.split == "random" else "") + ".json", "w"), indent=1)
print("POOLED", out["pooled"], out["ms_per_patch"])

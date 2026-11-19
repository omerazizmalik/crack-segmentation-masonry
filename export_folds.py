"""Write the fold assignment of every patch under both protocols to folds.csv."""
import sys, csv
sys.path.insert(0, ".")
from common import *

X, Y, G, N = load()
fg = {j: i for i, (_, te) in enumerate(folds(G)) for j in te}
fr = {j: i for i, (_, te) in enumerate(random_folds(len(G))) for j in te}
with open("folds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patch", "source_photograph", "test_fold_grouped", "test_fold_random"])
    for j, n in enumerate(N):
        w.writerow([n, G[j], fg[j], fr[j]])
print(f"{len(N)} patches from {len(set(G))} photographs written to folds.csv")

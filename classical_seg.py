"""Non-learned baseline: black-hat morphology + threshold + removal of small components (OpenCV)."""
import numpy as np, cv2

# Parameter grid searched on the training folds (structuring-element diameter k, threshold t, minimum area a)
GRID_K = (31, 41, 51, 61)
GRID_T = (75, 100, 125)
GRID_A = (100, 200, 400)


def segment(img, k, t, min_area):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = (bh > t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]

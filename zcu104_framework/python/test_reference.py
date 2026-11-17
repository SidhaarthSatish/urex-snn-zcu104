"""Checks the Python reference against (1) the dataset's ANF output and (2) the training patch generator."""
import numpy as np, sys, json
sys.path.insert(0, ".")
import paths
from pipeline import *
from snn_int import *
# 1. ANF on raw window, no mask, vs dataset truth (skip first 150 ms: unknown history)
r = np.load(paths.RAW_WINDOW); t = r["t"].astype(np.int64)
k = anf_batch(r["x"], r["y"], t); warm = t >= t[0] + 150000
print("ANF agreement:", (k[warm] == r["anf_truth"][warm]).mean(), "kept", int(k[warm].sum()), "truth", int(r["anf_truth"][warm].sum()))
assert (k[warm] == r["anf_truth"][warm]).all()
# effect of masking hot pixels BEFORE the ANF (proposed hardware order) vs after (training data order)
cfg = load_config("../config/pipeline.json")
m = hot_mask(r["x"], r["y"], cfg["hot_pixels"])
after = k & m                                       # training order: ANF then mask
kb = np.zeros(len(t), bool); kb[np.nonzero(m)[0]] = anf_batch(r["x"][m], r["y"][m], t[m])
print("mask-then-ANF vs ANF-then-mask (warm):", "kept", int(kb[warm].sum()), "vs", int(after[warm].sum()), "differ", int((kb != after)[warm].sum()))
# 2. patch builder vs ml/patches.make on the masked retained stream
import ml_patches as P0
d = np.load(paths.MASKED); x, y, p, tt = d["x"].astype(np.int32), d["y"].astype(np.int32), d["p"].astype(np.int8), d["t"].astype(np.int64)
rng = np.random.default_rng(1); idx = np.sort(rng.choice(len(x), 3000, replace=False))
Pm = P0.make(x, y, p, tt, idx, 16, 250000, 10)
bad = 0
for n, i in enumerate(idx):
    ev = patch_events(x, y, p, tt, i)
    if ev != patch_to_events(Pm[n]): bad += 1
print("patch builder mismatches:", bad, "of", len(idx)); assert bad == 0
print("ALL REFERENCE CHECKS PASSED")

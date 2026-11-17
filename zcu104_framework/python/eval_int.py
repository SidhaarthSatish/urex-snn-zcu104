import numpy as np, torch, json, sys
sys.path.insert(0, "."); from snn_int import *
import paths
ML = paths.MODELS
d = np.load(paths.P16); P = d["P"]; Y = d["y"]; Tt = d["t"]
blk = Tt // 60_000_000; s = (Tt - blk * 60_000_000) / 1e6
te = np.nonzero((blk % 5 == 4) & (s > 1))[0]
X = (P[te] > 0).reshape(len(te), T, NIN)
def prf(y, pr):
    tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum()
    a = tp / max(tp + fp, 1); b = tp / max(tp + fn, 1); return a, b, 2 * a * b / max(a + b, 1e-9)
# float reference (same as train.py FCSNN), recomputed here
def float_run(st, X, beta=0.8):
    W1 = st["f1.weight"].numpy().astype(np.float64); b1 = st["f1.bias"].numpy(); W2 = st["f2.weight"].numpy(); b2 = st["f2.bias"].numpy()
    m = np.zeros((len(X), NH)); out = np.zeros((len(X), 2))
    for k in range(T):
        m = beta * m + X[:, k] @ W1.T + b1; sp = (m > 1).astype(float); m -= sp; out += sp @ W2.T + b2
    out /= T; e = np.exp(out - out.max(1, keepdims=True)); return e[:, 1] / e.sum(1)
res = {}
for seed in (0, 1, 2):
    tag = f"fcsnn_p16_s{seed}"; r = json.load(open(f"{ML}/res_{tag}.json")); th = r["th"]
    st = torch.load(f"{ML}/w_{tag}.pt")
    pf = float_run(st, X); fl = prf(Y[te], pf >= th)
    Q = quantise(st, th); acc, mmax = run_batch(X, Q); pi = acc >= Q["ACC_TH"]
    it = prf(Y[te], pi)
    res[tag] = dict(float_f1=fl[2], int_f1=it[2], int_precision=it[0], int_recall=it[1], train_py_f1=r["f1"],
                    agreement=float(((pf >= th) == pi).mean()), max_abs_membrane=mmax, acc_range=[int(acc.min()), int(acc.max())],
                    TH_range=[int(Q["TH"].min()), int(Q["TH"].max())], B_range=[int(Q["B"].min()), int(Q["B"].max())], ACC_TH=Q["ACC_TH"])
    print(tag, json.dumps(res[tag]))
json.dump(res, open("int_eval.json", "w"), indent=1)
nz = (P[te] > 0).reshape(len(te), -1).sum(1); print("events per patch: mean %.1f p99 %d max %d" % (nz.mean(), np.percentile(nz, 99), nz.max()))

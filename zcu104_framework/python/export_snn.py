"""Export the trained FC-only SNN to integer form for the PS driver and the RTL testbench.

Outputs (in ../weights and ../sim/vectors):
  weights/snn_int8.json   W (512x128 int8), B, TH (int16), WD (int8), ACC_TH, metadata
  weights/snn_axi_writes.txt   "addr data" hex pairs: the exact AXI-Lite writes that load the core
  sim/vectors/wvec.mem    same writes for $readmemh (address, data alternating)
  sim/vectors/patches.mem golden patches: n, n x {bin[12:9], addr[8:0]}, expected acc, expected decision
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from snn_int import quantise, run_patch, patch_to_events, T, NIN, NH

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import paths
ML = paths.MODELS
TAG = "fcsnn_p16_s0"

# register map, must match rtl/snn_axi.v
REG_ACC_TH = 0x018
W1_BASE, B_BASE, TH_BASE, WD_BASE = 0x10000, 0x20000, 0x20400, 0x20800


def axi_writes(Q):
    w = []
    W = Q["W"].astype(np.int64) & 0xFF                     # (512, 128)
    for a in range(NIN):
        for q4 in range(NH // 4):
            j = q4 * 4
            word = W[a, j] | (W[a, j + 1] << 8) | (W[a, j + 2] << 16) | (W[a, j + 3] << 24)
            w.append((W1_BASE + 4 * (a * (NH // 4) + q4), int(word)))
    for j in range(NH):
        w.append((B_BASE + 4 * j, int(Q["B"][j]) & 0xFFFF))
        w.append((TH_BASE + 4 * j, int(Q["TH"][j]) & 0xFFFF))
        w.append((WD_BASE + 4 * j, int(Q["WD"][j]) & 0xFF))
    w.append((REG_ACC_TH, int(Q["ACC_TH"]) & 0xFFFFFFFF))
    return w


def main(n_random=1500):
    res = json.load(open(f"{ML}/res_{TAG}.json"))
    Q = quantise(torch.load(f"{ML}/w_{TAG}.pt"), res["th"])
    os.makedirs(f"{ROOT}/weights", exist_ok=True); os.makedirs(f"{ROOT}/sim/vectors", exist_ok=True)
    json.dump(dict(model=TAG, T=T, NIN=NIN, NH=NH, leak_num=205, leak_shift=8,
                   prob_threshold=res["th"], ACC_TH=Q["ACC_TH"],
                   W=Q["W"].tolist(), B=Q["B"].tolist(), TH=Q["TH"].tolist(), WD=Q["WD"].tolist(),
                   addr_layout="addr = pol*256 + ys*16 + xs; W[addr][neuron]",
                   test_f1_float=res["f1"]), open(f"{ROOT}/weights/snn_int8.json", "w"))
    wr = axi_writes(Q)
    with open(f"{ROOT}/weights/snn_axi_writes.txt", "w") as f:
        for a, d in wr: f.write(f"{a:08x} {d:08x}\n")
    with open(f"{ROOT}/sim/vectors/wvec.mem", "w") as f:
        for a, d in wr: f.write(f"{a:08x}\n{d:08x}\n")
    # golden patches from the held-out test split
    d = np.load(paths.P16); P = d["P"]; Tt = d["t"]
    blk = Tt // 60_000_000; s = (Tt - blk * 60_000_000) / 1e6
    te = np.nonzero((blk % 5 == 4) & (s > 1))[0]
    nz = (P[te] > 0).reshape(len(te), -1).sum(1)
    rng = np.random.default_rng(0)
    pick = list(rng.choice(te, n_random, replace=False)) + list(te[np.argsort(-nz)[:10]])
    words = []; n_patch = 0; maxev = 0
    patches = [[]] + [patch_to_events(P[i]) for i in pick]          # first one: empty patch
    for ev in patches:
        acc, dec, _ = run_patch(ev, Q)
        evs = sorted(ev)
        words.append(len(evs))
        words += [(b << 9) | a for b, a in evs]
        words.append(acc & 0xFFFFFFFF); words.append(dec)
        n_patch += 1; maxev = max(maxev, len(evs))
    with open(f"{ROOT}/sim/vectors/patches.mem", "w") as f:
        for w_ in words: f.write(f"{w_:08x}\n")
    with open(f"{ROOT}/sim/vectors/info.txt", "w") as f:
        f.write(f"patches {n_patch}\nwords {len(words)}\nweight_writes {len(wr)}\nmax_events {maxev}\nACC_TH {Q['ACC_TH']}\n")
    print(open(f"{ROOT}/sim/vectors/info.txt").read())


if __name__ == "__main__":
    main()

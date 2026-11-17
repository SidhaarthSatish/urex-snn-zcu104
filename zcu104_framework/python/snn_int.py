"""Integer (INT8 weight) FC-only SNN: the bit-exact reference the RTL must match.

Float model (train.py, FCSNN): for bin k = 0..T-1
    m = beta*m + W1 @ x_k + b1 ; s = (m > 1) ; m -= s ; out += W2 @ s + b2
Integer model, per hidden neuron j with its own scale q_j:
    m_j = leak(m_j) + B_j + sum_{active inputs a} W[a, j]      (int8 W, int16 B)
    s_j = m_j > TH_j ; if s_j: m_j -= TH_j                       (TH_j = round(1/q_j))
    acc += WD_j * s_j                                            (int8 WD = W2[1]-W2[0], quantised)
Decision: acc >= ACC_TH  (equivalent to softmax(out)[1] >= th of the float model).
leak(m) = (m * LEAK_NUM) >> LEAK_SHIFT  with arithmetic (floor) shift; 205/256 = 0.8008.
"""
import numpy as np, json, torch

T, NIN, NH = 10, 512, 128
LEAK_NUM, LEAK_SHIFT = 205, 8

def quantise(state, th_prob, T=T):
    W1 = state["f1.weight"].numpy().astype(np.float64)       # (128, 512)
    b1 = state["f1.bias"].numpy().astype(np.float64)
    W2 = state["f2.weight"].numpy().astype(np.float64)       # (2, 128)
    b2 = state["f2.bias"].numpy().astype(np.float64)
    q = np.abs(W1).max(1) / 127.0                            # per-neuron scale
    W = np.clip(np.round(W1 / q[:, None]), -127, 127).astype(np.int8).T.copy()   # (512, 128): row per input address
    B = np.clip(np.round(b1 / q), -32768, 32767).astype(np.int16)
    TH = np.round(1.0 / q).astype(np.int32)
    wd = W2[1] - W2[0]; bd = b2[1] - b2[0]
    qo = np.abs(wd).max() / 127.0
    WD = np.clip(np.round(wd / qo), -127, 127).astype(np.int8)
    # float decision: (sum_k wd.s_k)/T + bd >= logit(th)  ->  acc >= (logit(th) - bd) * T / qo
    logit = np.log(th_prob / (1 - th_prob))
    ACC_TH = int(np.ceil((logit - bd) * T / qo))
    return dict(W=W, B=B, TH=TH.astype(np.int16), WD=WD, ACC_TH=ACC_TH, qo=qo, bd=bd)

def leak(m):
    return (m * LEAK_NUM) >> LEAK_SHIFT

def run_patch(events, Q, trace=False):
    """events: list of (bin, addr) with bin in 0..T-1, addr = pol*256 + y*16 + x, any order.
    Returns (acc, decision, max|m|)."""
    m = np.zeros(NH, np.int64); acc = 0; mmax = 0
    ev = sorted(set((int(b), int(a)) for b, a in events))
    i = 0
    for k in range(T):
        m = leak(m) + Q["B"]
        while i < len(ev) and ev[i][0] == k:
            m = m + Q["W"][ev[i][1]]; i += 1
            mmax = max(mmax, int(np.abs(m).max()))
        s = m > Q["TH"]
        m = m - np.where(s, Q["TH"], 0)
        acc += int(Q["WD"][s].astype(np.int64).sum())
    return acc, int(acc >= Q["ACC_TH"]), mmax

def run_batch(X, Q):
    """Vectorised version of run_patch. X: (N, T, 512) binary. Returns acc (N,), mmax."""
    N = X.shape[0]; m = np.zeros((N, NH), np.int64); acc = np.zeros(N, np.int64)
    W = Q["W"].astype(np.int64); mmax = 0
    for k in range(T):
        m = leak(m) + Q["B"] + X[:, k].astype(np.int64) @ W
        mmax = max(mmax, int(np.abs(m).max()))
        s = m > Q["TH"]
        m = m - np.where(s, Q["TH"], 0)
        acc += s.astype(np.int64) @ Q["WD"].astype(np.int64)
    return acc, mmax

def patch_to_events(P):
    """P: (T, 2, 16, 16) counts -> list of (bin, addr) for nonzero voxels."""
    k, p, y, x = np.nonzero(P)
    return list(zip(k.tolist(), (p * 256 + y * 16 + x).tolist()))

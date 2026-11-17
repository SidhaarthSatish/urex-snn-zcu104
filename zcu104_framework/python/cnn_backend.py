"""Backup classifier: the stacked CNN, trained on the SAME binary patch format the SNN uses, so the two
back-ends are interchangeable behind the PS pipeline (event words -> dense 20x16x16 tensor).

    python3 cnn_backend.py train      # trains on ml/p16.npz with the same time-block split, writes weights/cnn_bin.pt
    python3 cnn_backend.py export     # ONNX (weights/cnn_fp32.onnx) + calibration set for Vitis AI + parity check

Deployment on the ZCU104 DPU (in the Vitis AI 3.x PyTorch docker; not run here):
    python3 vitis_ai/quantize_cnn.py --mode calib   then   --mode test   (writes quantize_result/*.xmodel)
    vai_c_xir -x quantize_result/UrexCNN_int.xmodel -a <arch.json of your DPU build> -o out -n urex_cnn
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paths
T, R = 10, 16


class UrexCNN(nn.Module):
    """Input (N, 20, 16, 16) float in {0,1}: channel = bin*2 + polarity."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(2 * T, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * (R // 2) ** 2, 128); self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))


def words_to_dense(ev_words):
    """Event words (bin << 9 | pol*256 + ys*16 + xs) -> (20, 16, 16) float32, channel = bin*2 + pol."""
    x = np.zeros((2 * T, R, R), np.float32)
    ev = np.asarray(ev_words, np.int64)
    if len(ev):
        b = ev >> 9; a = ev & 0x1FF; p = a >> 8; ys = (a >> 4) & 15; xs = a & 15
        x[b * 2 + p, ys, xs] = 1.0
    return x


def split():
    d = np.load(paths.P16); P = d["P"]; Y = d["y"]; Tt = d["t"]
    blk = Tt // 60_000_000; s = (Tt - blk * 60_000_000) / 1e6
    role = np.where(blk % 5 == 4, 2, np.where(blk % 5 == 3, 1, 0)); ok = (role == 0) | (s > 1)
    X = (P > 0).astype(np.float32).reshape(len(P), 2 * T, R, R)          # (N, T, 2, R, R) -> channel = bin*2 + pol
    return X, Y, [np.nonzero((role == r) & ok)[0] for r in (0, 1, 2)]


def prf(y, pr):
    tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum()
    a = tp / max(tp + fp, 1); b = tp / max(tp + fn, 1); return float(a), float(b), float(2 * a * b / max(a + b, 1e-9))


def predict(M, X):
    M.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.softmax(M(torch.from_numpy(X[i:i + 1024])), 1)[:, 1].numpy())
    return np.concatenate(out)


def train(epochs=6, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    X, Y, (tr, va, te) = split(); M = UrexCNN(); opt = torch.optim.Adam(M.parameters(), lr=2e-3)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y).long(); best = (-1, None); t0 = time.time()
    from sklearn.metrics import average_precision_score
    for ep in range(epochs):
        M.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]; loss = F.cross_entropy(M(Xt[b]), Yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        ap = average_precision_score(Y[va], predict(M, X[va])); print(f"ep{ep} val AP {ap:.4f} {time.time()-t0:.0f}s", flush=True)
        if ap > best[0]: best = (ap, {k: v.clone() for k, v in M.state_dict().items()})
    M.load_state_dict(best[1])
    pv = predict(M, X[va]); ths = np.linspace(0.05, 0.95, 91)
    th = float(ths[np.argmax([prf(Y[va], pv >= t)[2] for t in ths])])
    p, r, f = prf(Y[te], predict(M, X[te]) >= th)
    res = dict(model="cnn_binary_input", precision=p, recall=r, f1=f, th=th, seed=seed, test_n=int(len(te)))
    os.makedirs(f"{ROOT}/weights", exist_ok=True)
    torch.save(M.state_dict(), f"{ROOT}/weights/cnn_bin.pt"); json.dump(res, open(f"{ROOT}/weights/cnn_bin.json", "w"), indent=1)
    print(json.dumps(res))


def export():
    import onnxruntime as ort
    M = UrexCNN(); M.load_state_dict(torch.load(f"{ROOT}/weights/cnn_bin.pt")); M.eval()
    X, Y, (tr, va, te) = split()
    dummy = torch.zeros(1, 2 * T, R, R)
    torch.onnx.export(M, dummy, f"{ROOT}/weights/cnn_fp32.onnx", input_names=["patch"], output_names=["logits"],
                      dynamic_axes={"patch": {0: "n"}, "logits": {0: "n"}}, opset_version=13, dynamo=False)
    s = ort.InferenceSession(f"{ROOT}/weights/cnn_fp32.onnx")
    xb = X[te[:2000]]
    o1 = s.run(None, {"patch": xb})[0]
    with torch.no_grad(): o2 = M(torch.from_numpy(xb)).numpy()
    print("ONNX vs torch max |diff|:", float(np.abs(o1 - o2).max()))
    rng = np.random.default_rng(0)
    np.savez_compressed(f"{ROOT}/weights/cnn_calib.npz", x=X[rng.choice(tr, 1000, replace=False)])
    np.savez_compressed(f"{ROOT}/weights/cnn_test.npz", x=X[te], y=Y[te])
    print("wrote cnn_fp32.onnx, cnn_calib.npz (1000 train patches), cnn_test.npz")


if __name__ == "__main__":
    {"train": train, "export": export}[sys.argv[1]]()

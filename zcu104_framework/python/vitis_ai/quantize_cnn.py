"""Vitis AI (3.x, PyTorch flow) quantisation of the backup CNN. Run INSIDE the Vitis AI PyTorch docker:
    python3 quantize_cnn.py --mode calib
    python3 quantize_cnn.py --mode test          # reports INT8 F1 and writes quantize_result/UrexCNN_int.xmodel
    vai_c_xir -x quantize_result/UrexCNN_int.xmodel -a <arch.json from your DPU build> -o out -n urex_cnn
NOT YET RUN: written without access to the Vitis AI docker.
"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from cnn_backend import UrexCNN
from pytorch_nndct.apis import torch_quantizer

ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["calib", "test"], required=True); a = ap.parse_args()
M = UrexCNN(); M.load_state_dict(torch.load(f"{ROOT}/weights/cnn_bin.pt", map_location="cpu")); M.eval()
th = json.load(open(f"{ROOT}/weights/cnn_bin.json"))["th"]
dummy = torch.zeros(1, 20, 16, 16)
q = torch_quantizer(a.mode, M, (dummy,), output_dir="quantize_result")
qm = q.quant_model
if a.mode == "calib":
    x = np.load(f"{ROOT}/weights/cnn_calib.npz")["x"]
    with torch.no_grad():
        for i in range(0, len(x), 100): qm(torch.from_numpy(x[i:i + 100]))
    q.export_quant_config()
else:
    d = np.load(f"{ROOT}/weights/cnn_test.npz"); x, y = d["x"], d["y"]; p = []
    with torch.no_grad():
        for i in range(0, len(x), 500): p.append(torch.softmax(qm(torch.from_numpy(x[i:i + 500])), 1)[:, 1].numpy())
    pr = np.concatenate(p) >= th; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum()
    P_, R_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1); print("INT8 test F1:", 2 * P_ * R_ / max(P_ + R_, 1e-9))
    with torch.no_grad(): qm(dummy)                      # export needs a batch-1 forward
    q.export_xmodel(deploy_check=False)

"""One interface, four back-ends. Every back-end takes the same input: a sorted uint16 array of event words
(bin << 9 | pol*256 + ys*16 + xs) from the PS pipeline, and returns (positive: 0/1, score).

    clf = make_classifier(cfg, root)       # cfg["classifier"]["backend"] in {"snn", "snn_cpu", "cnn_dpu", "cnn_cpu"}
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "python"))


class SnnPL:                       # main path: SNN core in the PL
    def __init__(self, root, bit=None, **_):
        from snn_driver import SnnAccel
        self.hw = SnnAccel(bit or os.path.join(root, "vivado/out/urex_snn.bit"))
        self.hw.load_weights(os.path.join(root, "weights/snn_int8.json"))
    def __call__(self, ev):
        acc, d, _ = self.hw.classify(ev); return d, acc


class SnnCPU:                      # same INT8 SNN on the A53 (baseline / fallback)
    def __init__(self, root, **_):
        from urex import SoftSNN
        self.s = SoftSNN(os.path.join(root, "weights/snn_int8.json"))
    def __call__(self, ev):
        acc, d = self.s.classify(ev); return d, acc


class CnnCPU:                      # backup CNN through ONNX Runtime (no DPU needed)
    def __init__(self, root, **_):
        import onnxruntime as ort
        from cnn_backend import words_to_dense
        self.s = ort.InferenceSession(os.path.join(root, "weights/cnn_fp32.onnx")); self.f = words_to_dense
        self.th = json.load(open(os.path.join(root, "weights/cnn_bin.json")))["th"]
    def __call__(self, ev):
        z = self.s.run(None, {"patch": self.f(ev)[None]})[0][0]
        p = float(np.exp(z[1]) / np.exp(z).sum()); return int(p >= self.th), p


class CnnDPU:                      # backup CNN on the DPU (needs a DPU overlay + compiled xmodel). NOT YET RUN.
    def __init__(self, root, dpu_bit="dpu.bit", xmodel=None, **_):
        from pynq_dpu import DpuOverlay
        from cnn_backend import words_to_dense
        self.ov = DpuOverlay(dpu_bit); self.ov.load_model(xmodel or os.path.join(root, "weights/urex_cnn.xmodel"))
        self.r = self.ov.runner; self.f = words_to_dense
        it, ot = self.r.get_input_tensors()[0], self.r.get_output_tensors()[0]
        self.ishape, self.oshape = tuple(it.dims), tuple(ot.dims)          # expected (1, 16, 16, 20) NHWC
        self.th = json.load(open(os.path.join(root, "weights/cnn_bin.json")))["th"]
    def __call__(self, ev):
        x = np.ascontiguousarray(self.f(ev).transpose(1, 2, 0)[None], np.float32).reshape(self.ishape)
        y = np.empty(self.oshape, np.float32)
        jid = self.r.execute_async([x], [y]); self.r.wait(jid)
        z = y.reshape(-1); p = float(np.exp(z[1]) / np.exp(z).sum()); return int(p >= self.th), p


BACKENDS = {"snn": SnnPL, "snn_cpu": SnnCPU, "cnn_cpu": CnnCPU, "cnn_dpu": CnnDPU}


def make_classifier(cfg, root=ROOT, **kw):
    return BACKENDS[cfg["classifier"]["backend"]](root, **kw)

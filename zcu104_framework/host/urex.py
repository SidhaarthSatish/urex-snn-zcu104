"""ctypes wrapper for urex_pipe.c (mask + ANF + patch builder + software SNN). Works on x86 and on the ZCU104 (aarch64)."""
import ctypes as C, json, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "liburex_pipe.so")


def build(force=False):
    src = os.path.join(HERE, "urex_pipe.c")
    if force or not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", LIB, src])
    return C.CDLL(LIB)


_lib = build()
_lib.urex_create.restype = C.c_void_p
_lib.urex_create.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int64, C.c_int, C.c_int64, C.c_int]
_lib.urex_destroy.argtypes = [C.c_void_p]
_lib.urex_run.restype = C.c_int64
_lib.urex_run.argtypes = [C.c_void_p, C.c_int64] + [C.c_void_p] * 4 + [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64]
_lib.urex_snn.restype = C.c_int32
_lib.urex_snn.argtypes = [C.c_void_p] * 4 + [C.c_int] * 5 + [C.c_void_p, C.c_int]


class Pipeline:
    """Streaming PS pipeline. State persists across run() calls, so feed the stream in chunks."""

    def __init__(self, cfg):
        hot = np.array(cfg["hot_pixels"], np.int32).reshape(-1)
        a = cfg["anf"]; p = cfg["patch"]
        self.cfg = cfg
        self.ctx = _lib.urex_create(hot.ctypes.data_as(C.POINTER(C.c_int)), len(hot) // 2, int(a["enabled"]),
                                    a["threshold_us"], p["R"], p["window_us"], p["bins"])

    def run(self, x, y, p, t, ev_cap=None):
        n = len(x)
        x = np.ascontiguousarray(x, np.uint16); y = np.ascontiguousarray(y, np.uint16)
        p = np.ascontiguousarray(p, np.uint8); t = np.ascontiguousarray(t, np.int64)
        status = np.zeros(n, np.int8); off = np.zeros(n + 1, np.int64)
        cap = ev_cap or max(1 << 20, 64 * n)
        while True:
            buf = np.zeros(cap, np.uint16)
            r = _lib.urex_run(self.ctx, n, x.ctypes.data, y.ctypes.data, p.ctypes.data, t.ctypes.data,
                              status.ctypes.data, buf.ctypes.data, off.ctypes.data, cap)
            if r >= 0:
                return status, buf[:r], off
            raise RuntimeError("event buffer too small; pass a larger ev_cap (state already advanced)")

    def __del__(self):
        if getattr(self, "ctx", None):
            _lib.urex_destroy(self.ctx); self.ctx = None


class SoftSNN:
    """CPU INT8 SNN, same arithmetic as the RTL core. Used as the PS-only baseline."""

    def __init__(self, weights_json):
        w = json.load(open(weights_json))
        self.W = np.ascontiguousarray(np.array(w["W"], np.int8))
        self.B = np.ascontiguousarray(np.array(w["B"], np.int16))
        self.TH = np.ascontiguousarray(np.array(w["TH"], np.int16))
        self.WD = np.ascontiguousarray(np.array(w["WD"], np.int8))
        self.NIN, self.NH, self.T = w["NIN"], w["NH"], w["T"]
        self.leak_num, self.leak_shift, self.ACC_TH = w["leak_num"], w["leak_shift"], w["ACC_TH"]

    def score(self, ev_words):
        ev = np.ascontiguousarray(ev_words, np.uint16)
        return int(_lib.urex_snn(self.W.ctypes.data, self.B.ctypes.data, self.TH.ctypes.data, self.WD.ctypes.data,
                                 self.NIN, self.NH, self.T, self.leak_num, self.leak_shift, ev.ctypes.data, len(ev)))

    def classify(self, ev_words):
        a = self.score(ev_words)
        return a, int(a >= self.ACC_TH)

"""C pipeline + software SNN vs the Python reference (run on x86 here, and on the board as a self-test)."""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "python"))
from urex import Pipeline, SoftSNN
from pipeline import load_config, run_stream, patch_events

import paths
cfg = load_config(os.path.join(ROOT, "config/pipeline.json"))
# 1. raw window: mask + ANF + patches, C vs Python
r = np.load(paths.RAW_WINDOW); x, y, p, t = r["x"], r["y"], r["p"], r["t"].astype(np.int64)
t0 = time.time(); st, buf, off = Pipeline(cfg).run(x, y, p, t); tc = time.time() - t0
raw_idx, xr, yr, pr, tr = run_stream(x, y, p, t, cfg)
ret_c = np.nonzero(st == 1)[0]
print(f"C pipeline: {len(x):,} raw events in {tc:.2f} s ({len(x)/tc/1e6:.1f} M ev/s); retained {len(ret_c):,}, python {len(raw_idx):,}")
assert np.array_equal(ret_c, raw_idx), "retained sets differ"
rng = np.random.default_rng(0); bad = 0
for k in rng.choice(len(raw_idx), 3000, replace=False):
    ref = [(b << 9) | a for b, a in patch_events(xr, yr, pr, tr, k)]
    i = raw_idx[k]
    if list(buf[off[i]:off[i + 1]]) != ref: bad += 1
print("patch mismatches:", bad, "of 3000"); assert bad == 0
# 2. software SNN vs golden vectors
snn = SoftSNN(os.path.join(ROOT, "weights/snn_int8.json"))
v = [int(l, 16) for l in open(os.path.join(ROOT, "sim/vectors/patches.mem"))]
pos = 0; n_p = 0; bad = 0
while pos < len(v):
    n = v[pos]; ev = np.array(v[pos + 1:pos + 1 + n], np.uint16); pos += 1 + n
    exp = v[pos] - (1 << 32) if v[pos] >= (1 << 31) else v[pos]; dec = v[pos + 1]; pos += 2
    a, d = snn.classify(ev); bad += (a != exp) or (d != dec); n_p += 1
print("software SNN mismatches:", bad, "of", n_p); assert bad == 0
print("ALL HOST CHECKS PASSED")

"""Reference event pipeline: hot-pixel mask -> ANF -> 16x16 patch builder.

This is the golden model for the C (PS) and RTL (PL) implementations. Every stage
is defined on an event stream in arrival order; "earlier" always means earlier
index in the stream, so events with equal timestamps are handled deterministically.

ANF definition (verified bit-exact against the SBSS release's ANF membership on a
3.9 M-event window, 1,155,957 kept events, zero mismatches):
    an event is kept if any of the 28 pixels with 0 < dx^2 + dy^2 <= 9 had an
    event (kept or not, any polarity) at an earlier index with t_i - t_n <= 150000 us.
    The per-pixel timestamp map is updated by every event that reaches the filter.

Patch definition (identical to ml/patches.py, which produced the training data):
    for retained event i, take retained events j <= i with t_j >= t_i - W,
    |box|: xs = x_j - x_i + R/2, ys = y_j - y_i + R/2 in [0, R),
    bin = (t_j - (t_i - W)) * T // (W + 1), voxel (bin, p_j, ys, xs) set to 1.
    Address sent to the classifier: addr = p*R*R + ys*R + xs.
"""
import json
import numpy as np

H, W_SENSOR = 720, 1280
ANF_OFFSETS = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if 0 < dx * dx + dy * dy <= 9]
assert len(ANF_OFFSETS) == 28


def load_config(path):
    with open(path) as f:
        return json.load(f)


def hot_mask(x, y, hot_pixels):
    """True for events that survive the mask. hot_pixels: list of [x, y]."""
    keep = np.ones(len(x), bool)
    for hx, hy in hot_pixels:
        keep &= ~((x == hx) & (y == hy))
    return keep


def anf_batch(x, y, t, radius_offsets=ANF_OFFSETS, thr_us=150000):
    """Vectorised exact ANF over a whole stream. Returns kept mask.
    The stream must start with an empty timestamp map (no history)."""
    x = np.asarray(x, np.int64); y = np.asarray(y, np.int64); t = np.asarray(t, np.int64)
    N = len(x); idx = np.arange(N); pix = x * H + y
    order = np.lexsort((idx, pix)); keys = pix[order] * N + idx[order]
    best = np.full(N, np.iinfo(np.int64).max)
    for dx, dy in radius_offsets:
        nx = x + dx; ny = y + dy
        ok = (nx >= 0) & (nx < W_SENSOR) & (ny >= 0) & (ny < H)
        npix = nx * H + ny
        j = np.searchsorted(keys, npix * N + idx) - 1
        jj = np.maximum(j, 0)
        found = ok & (j >= 0) & (keys[jj] // N == npix)
        best = np.minimum(best, np.where(found, t - t[order[jj]], np.iinfo(np.int64).max))
    return best <= thr_us


def patch_events(x, y, p, t, i, R=16, W_us=250000, T=10):
    """Sorted unique (bin, addr) list for retained event i of a retained stream (arrays)."""
    lo = np.searchsorted(t, t[i] - W_us, side="left")
    sl = slice(lo, i + 1); h = R // 2
    xs = x[sl].astype(np.int64) - int(x[i]) + h
    ys = y[sl].astype(np.int64) - int(y[i]) + h
    m = (xs >= 0) & (xs < R) & (ys >= 0) & (ys < R)
    b = ((t[sl][m].astype(np.int64) - (int(t[i]) - W_us)) * T // (W_us + 1)).clip(0, T - 1)
    addr = p[sl][m].astype(np.int64) * R * R + ys[m] * R + xs[m]
    return sorted(set(zip(b.tolist(), addr.tolist())))


def run_stream(x, y, p, t, cfg):
    """Full reference: raw stream -> (retained indices into raw stream, retained arrays)."""
    keep = hot_mask(x, y, cfg["hot_pixels"])
    xi, yi, pi, ti = x[keep], y[keep], p[keep], t[keep]
    k = anf_batch(xi, yi, ti, thr_us=cfg["anf"]["threshold_us"]) if cfg["anf"]["enabled"] else np.ones(len(xi), bool)
    raw_idx = np.nonzero(keep)[0][k]
    return raw_idx, xi[k], yi[k], pi[k], ti[k]

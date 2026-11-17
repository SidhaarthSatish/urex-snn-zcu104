"""Board benchmark for the ZCU104 (run on the board under PYNQ, as root):

    sudo python3 bench.py --bit ../vivado/out/urex_snn.bit --raw raw_window_b64.npz

Steps:
  1. bit-exact check: every golden patch in sim/vectors/patches.mem through the PL core
  2. latency: PL compute time (cycles / f_clk), PL round trip incl. AXI-Lite writes, and the
     CPU-only INT8 SNN on the A53, per patch
  3. end-to-end: raw events -> C mask/ANF/patch builder -> PL SNN, events per second
  4. power: board rails sampled idle and while classifying (needs PYNQ rail sensors)
Results go to bench_results.json. NOT YET RUN ON A BOARD.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "python"))
from snn_driver import SnnAccel
from urex import Pipeline, SoftSNN


def golden(path):
    v = [int(l, 16) for l in open(path)]; pos = 0; out = []
    while pos < len(v):
        n = v[pos]; ev = np.array(v[pos + 1:pos + 1 + n], np.uint16); pos += 1 + n
        a = v[pos] - (1 << 32) if v[pos] >= (1 << 31) else v[pos]; out.append((ev, a, v[pos + 1])); pos += 2
    return out


def rails_sample(seconds):
    """Mean power per rail over `seconds`, or None if PYNQ rail sensors are unavailable."""
    try:
        import pynq
        rails = pynq.get_rails()
        rec = pynq.DataRecorder(*[r.power for r in rails.values()])
        with rec.record(0.05):
            time.sleep(seconds)
        return {k: float(rec.frame[c].mean()) for k, c in zip(rails, rec.frame.columns)}
    except Exception as e:                                       # noqa
        return {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", default=os.path.join(ROOT, "vivado/out/urex_snn.bit"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights/snn_int8.json"))
    ap.add_argument("--vectors", default=os.path.join(ROOT, "sim/vectors/patches.mem"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "config/pipeline.json"))
    ap.add_argument("--raw", default=None, help="npz with x, y, p, t raw events (e.g. raw_window_b64.npz)")
    ap.add_argument("--clk_mhz", type=float, default=200.0)
    a = ap.parse_args()
    res = {}
    hw = SnnAccel(a.bit); hw.load_weights(a.weights); sw = SoftSNN(a.weights)
    res["core"] = dict(lanes=hw.lanes, T=hw.T, NH=hw.NH, clk_mhz=a.clk_mhz)

    g = golden(a.vectors); bad = 0; cyc = []; t_hw = []; t_sw = []
    for ev, exp, dec in g:
        t0 = time.perf_counter(); acc, d, c = hw.classify(ev); t_hw.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); sw.score(ev); t_sw.append(time.perf_counter() - t0)
        bad += (acc != exp) or (d != dec); cyc.append(c)
    cyc = np.array(cyc)
    res["bit_exact"] = dict(patches=len(g), mismatches=int(bad))
    res["latency_us"] = dict(
        pl_compute_mean=float(cyc.mean() / a.clk_mhz), pl_compute_max=float(cyc.max() / a.clk_mhz),
        pl_round_trip_python_mean=float(np.mean(t_hw) * 1e6),
        a53_int8_snn_mean=float(np.mean(t_sw) * 1e6))
    print(json.dumps(res, indent=1))

    if a.raw:
        from pipeline import load_config
        cfg = load_config(a.cfg); r = np.load(a.raw)
        pipe = Pipeline(cfg)
        t0 = time.perf_counter(); st, buf, off = pipe.run(r["x"], r["y"], r["p"], r["t"]); t_pipe = time.perf_counter() - t0
        idx = np.nonzero(st == 1)[0]
        t0 = time.perf_counter(); pos = 0
        for i in idx:
            _, d, _ = hw.classify(buf[off[i]:off[i + 1]]); pos += d
        t_cls = time.perf_counter() - t0
        res["end_to_end"] = dict(raw_events=int(len(st)), retained=int(len(idx)), positives=int(pos),
                                 ps_pipeline_s=t_pipe, ps_pipeline_Mev_s=len(st) / t_pipe / 1e6,
                                 classify_s=t_cls, patches_per_s=len(idx) / t_cls)

    res["power_idle_W"] = rails_sample(5.0)
    # busy loop on the PL in a background thread while sampling
    import threading
    stop = [False]
    def spin():
        while not stop[0]:
            for ev, _, _ in g[:200]:
                hw.classify(ev)
    th = threading.Thread(target=spin); th.start()
    res["power_busy_W"] = rails_sample(5.0); stop[0] = True; th.join()
    json.dump(res, open("bench_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

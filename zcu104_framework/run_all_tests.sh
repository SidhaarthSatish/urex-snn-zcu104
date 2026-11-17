#!/bin/bash
# Re-runs every offline check. Needs: python3 with numpy, torch, scikit-learn, onnxruntime; gcc; iverilog.
# Data paths (ml/p16.npz, masked_events.npz, raw_window_b64.npz) are set at the top of each test script.
set -e
cd "$(dirname "$0")"
echo "== 1. integer SNN vs float (3 seeds)";            (cd python && python3 eval_int.py | tail -4)
echo "== 2. reference pipeline vs dataset ANF + training patches"; (cd python && python3 test_reference.py | tail -1)
echo "== 3. export weights + golden vectors";           (cd python && python3 export_snn.py | head -2)
echo "== 4. RTL simulation, bit-exact (LANES 16 and 128)"
for L in 16 128; do
  (cd sim && iverilog -g2012 -P tb_snn_axi.LANES=$L -o tb_$L.vvp tb_snn_axi.v ../rtl/snn_axi.v ../rtl/snn_core.v && vvp -n tb_$L.vvp | grep -E "RESULT|PASS|FAIL")
done
echo "== 5. C pipeline + software SNN";                  (cd host && python3 test_host.py | tail -1)
echo "== 6. PYNQ driver write sequence (offline)";      (cd host && python3 test_driver_offline.py | head -1)
echo "ALL DONE"

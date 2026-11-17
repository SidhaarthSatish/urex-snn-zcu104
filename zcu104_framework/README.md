# UREX ZCU104 framework: event-based RSO detection with an FC-only SNN

Pipeline (one event stream, in arrival order):

```
IMX636 raw events ─► hot-pixel mask (7 px) ─► ANF noise filter ─► 16×16 patch builder ─► classifier ─► streak / not
                     └──────────── PS (A53, C: host/urex_pipe.c) ──────────────────┘   └ PL: rtl/snn_core.v (main)
                                                                                       └ DPU: stacked CNN (backup)
```

| Stage | Definition |
|---|---|
| Hot mask | 7 pixels listed in `config/pipeline.json`. Six of them produce 99.94% of the 21.9 B raw events. |
| ANF | Keep the event if any of the 28 pixels with 0 < dx²+dy² ≤ 9 had an event within ≤ 150,000 µs. The map is updated by every event that passes the mask. |
| Patch | Retained events in the preceding 250 ms inside a 16×16 box, in 10 time bins × 2 polarities, as binary voxels. Each voxel is sent as a 13-bit word `bin<<9 \| pol*256 + ys*16 + xs`. |
| SNN | 512 inputs → 128 LIF neurons → 1 score. INT8 weights, leak = ×205 >> 8. Decision: score ≥ ACC_TH. |
| CNN (backup) | Conv 20→16, Conv 16→32 + max-pool, FC 2048→128, FC 128→2. Same binary patch input. |

## Status

### Verified offline (`./run_all_tests.sh`)

| Check | Result |
|---|---|
| ANF definition against the SBSS release's own ANF output (3.9 M raw events) | Exact: 2,513,822 of 2,513,822 decisions match |
| Mask before ANF instead of after (training-data order) | 4 of 113,659 retained events differ |
| Patch builder against the generator used for training | 0 mismatches in 3,000 |
| INT8 SNN against float, 3 seeds | F1 0.9531 / 0.9527 / 0.9525 against float 0.9529 / 0.9532 / 0.9523; 99.9% identical decisions |
| RTL (`snn_axi` + `snn_core`) in iverilog against the Python integer model, LANES = 4, 16, 32, 128 | 1,511 of 1,511 patches bit-exact, including an empty patch and the 10 largest (up to 670 events) |
| C pipeline (`urex_pipe.c`) against the Python reference | Identical retained set (153,435) and patches; 10.5 M raw events/s on one x86 core |
| C software SNN against golden vectors | 1,511 of 1,511 |
| PYNQ driver weight-load sequence against exported AXI writes | 16,769 of 16,769 writes identical |
| Backup CNN with binary input | F1 0.9575 on the same split; ONNX matches PyTorch to 1e-6 |

### Not yet done (needs Vivado, Vitis AI or the board)

- **Vivado, full chip.** `synth_ooc.tcl` (core only) has run in Vivado 2018.2. `build.tcl` fails on this PC because 2018.2 cannot resolve a module reference ("Unable to resolve module-source"). `build2.tcl` + `rtl/urex_top.v` is the workaround (RTL top outside the block design). It builds a full bitstream; results below. A newer Vivado (2019.2+) should accept `build.tcl` unchanged.
- **Board.** `host/bench.py` and `host/snn_driver.py` have not been run on a ZCU104.
- **Vitis AI.** `python/vitis_ai/quantize_cnn.py` and the DPU back-end have not been run.
- **Part number.** The ZCU104 part is assumed to be `xczu7ev-ffvc1156-2-e`, with 230,400 LUTs, 460,800 FFs, 1,728 DSPs, 312 BRAM36 and 96 URAM. Confirm these against UG1267 and DS891.

### Vivado results (xczu7ev-ffvc1156-2-e, Vivado 2018.2, core out-of-context, place + route)

| | LANES = 16, 200 MHz |
|---|---|
| CLB LUTs | 8,413 (3.65 %), of which 674 are distributed RAM |
| Flip-flops | 3,378 (0.73 %) |
| DSP48E2 | 16 (the 16 leak multipliers) |
| Block RAM | 19 RAMB36 (16 for the W1 weights, 3 for the event FIFO) |
| Timing | met: WNS +0.693 ns, WHS +0.065 ns |
| Cycles per patch (RTL simulation, mean over 1,511 golden patches) | 334, so 1.67 µs at 200 MHz |

Power has not been measured yet. Run `report_power` on the full-chip build and then measure the rails on the board.

### Full-chip build (`build2.tcl`, 16 lanes, 150 MHz, PS + interconnect + core)

| | |
|---|---|
| CLB LUTs / FFs | 9,569 (4.15 %) / 4,614 (1.00 %) |
| DSP48E2 / RAMB36 | 16 / 19 |
| Timing | met: WNS +1.812 ns, WHS +0.012 ns (hold slack is thin but positive) |
| Vectorless power (Vivado `report_power`, confidence Medium) | 3.57 W total: 2.88 W dynamic + 0.69 W static. Most of the dynamic part is the Zynq PS, not the SNN core. A vectorless number says nothing about the SNN's own energy per patch. Measure the rails on the board. |
| Outputs | `vivado/out2/urex_snn.bit`, `.hwh`, `.hdf`, `utilisation*.rpt`, `timing.rpt`, `power_vectorless.rpt` |

This was built at 150 MHz; the core alone met 200 MHz out-of-context. Re-run `build2.tcl 16 200` to confirm 200 MHz in the full chip. The `.bit` has not been loaded on a board.

**Two things that cost time, so they are not repeated:**
1. The first 250 MHz run failed timing (WNS -5.9 ns) because the fire-and-sum logic was one combinational path. The core now has a 3-stage score pipeline and a pipelined leak, and meets 200 MHz.
2. Vivado 2018.2 silently synthesised only lane 0 of the neuron array when the fire logic was written as a `for` loop with shared blocking temporaries in an `always @*`. The result looked fine (1,084 LUTs, timing met) but was wrong: 15 of 16 lanes had been trimmed. iverilog, Verilator and Yosys all accepted that code. It is now one `generate` block per lane. **Always check the log for "Synth 8-3332 ... is unused and will be removed" on `mem_reg`, and check that `mem_reg` has LANES x MEMW x G = 2,560 flip-flops** (`vivado/dbg_synth.tcl` prints the survivors). Simulation can never catch this class of problem.

### Estimates (not measurements)

| | LANES = 16 | LANES = 128 |
|---|---|---|
| Cycles per patch (RTL simulation, before the pipelining change for 128) | 334 | about 62 |
| Time per patch at 200 MHz | 1.67 µs | about 0.31 µs |

The cycle count is about `T*(2*G+3) + n_events*G`, where G = 128/LANES. The average retained rate after the mask is 863 events per second, and the busiest second has 361,440. So even LANES = 16 leaves headroom. On the first board run, the bottleneck will be the Python AXI-Lite writes, not the core.

## Folder layout

```
config/pipeline.json      hot pixels, ANF, patch and classifier settings (backend: snn | snn_cpu | cnn_cpu | cnn_dpu)
python/                   reference model and training
  pipeline.py             golden mask / ANF / patch builder
  snn_int.py              golden integer SNN (the RTL must match this)
  export_snn.py           writes weights/snn_int8.json, AXI write list and sim/vectors
  eval_int.py             INT8 vs float F1
  cnn_backend.py          binary-input CNN: train, export ONNX and calibration data
  vitis_ai/quantize_cnn.py  Vitis AI INT8 quantisation and xmodel export (run inside the docker)
  train_models.py, ml_patches.py, baselines.py   the scripts that produced the CNN/CSNN/SNN comparison
  test_reference.py       checks against the dataset ANF and the training patches
rtl/snn_core.v            event-driven LIF core (LANES parameter), runtime-loadable weights, 8192-event FIFO
rtl/snn_axi.v             AXI4-Lite slave + done interrupt (register map in the header)
sim/tb_snn_axi.v          loads weights over AXI, runs all golden patches, checks bit-exactness, reports cycles
vivado/build.tcl          ZCU104 block design (module reference): PS + SmartConnect + snn_axi at 0xA000_0000, IRQ to pl_ps_irq0
vivado/build2.tcl         same system for Vivado 2018.2: block design with PS + interconnect, RTL top rtl/urex_top.v around it
vivado/dbg_synth.tcl      synthesis only; lists which registers survived (catches silent trimming)
vivado/synth_ooc.tcl      out-of-context synthesis of the core for quick utilisation and timing numbers
host/urex_pipe.c          PS pipeline in C (mask, ANF, patch builder) + software INT8 SNN
host/urex.py              ctypes wrapper (Pipeline, SoftSNN)
host/snn_driver.py        PYNQ MMIO driver for the PL core
host/classifier.py        one interface for all four back-ends
host/bench.py             on-board: bit-exact check, latency, end-to-end throughput, rail power
weights/                  snn_int8.json, snn_axi_writes.txt, cnn_bin.pt, cnn_fp32.onnx, calibration and test sets
data/                     p16.npz (100k patches), raw_window_b64.npz (3.9 M raw events + ANF truth), models/
```

## Bring-up on the ZCU104

1. **Build the bitstream** on a PC with Vivado 2019.2 or newer and the ZCU104 board files:
   `cd vivado && vivado -mode batch -source build.tcl -tclargs 16 200`
   This produces `out/urex_snn.bit`, `.hwh`, `.xsa` and the reports. If timing fails at 200 MHz, rebuild at 150. For core-only numbers, run `synth_ooc.tcl` first.
   On Vivado 2018.2 use `build2.tcl` instead (`-tclargs 16 150`); it writes to `out2/`.
2. **Prepare the board.** Boot a PYNQ image for the ZCU104 and copy this folder to it. Then run `cd host && python3 -c "import urex"`, which compiles `liburex_pipe.so` for aarch64.
3. **Self-test without the PL:** `python3 test_host.py`. This needs `data/` and the masked-events file (see `python/paths.py`).
4. **Benchmark the PL:**
   `sudo python3 bench.py --bit ../vivado/out/urex_snn.bit --raw ../data/raw_window_b64.npz --clk_mhz 200`
   - `bit_exact.mismatches` must be 0.
   - Record `latency_us`, `end_to_end` and `power_*_W` in `bench_results.json`.
5. **Run the CNN backup:**
   - Inside the Vitis AI docker, run `python3 python/vitis_ai/quantize_cnn.py --mode calib`, then `--mode test`.
   - Compile with `vai_c_xir` against the `arch.json` of the DPU you deploy.
   - On the board, set `"backend": "cnn_dpu"` in the config.

## Next steps (in order)

1. **Full-chip Vivado build.** Core-only numbers are in. Record full-chip utilisation, timing and vectorless power from `build2.tcl`.
2. **First board run.** Get bit-exact on hardware, then measure latency and power with `bench.py`, against the A53 software SNN and the CNN.
3. **Faster host interface.** Replace the per-event AXI-Lite writes with an AXI-Stream event port and AXI DMA, or at least C-level MMIO.
4. **Move the mask and ANF into the PL.** The ANF needs a per-pixel timestamp map: 921,600 pixels × 18-bit µs timestamps (262 ms wrap, plus a scrub) ≈ 16.6 Mb ≈ 57 of 96 URAM. Check this against the PS throughput measured on the A53 first.
5. **Data.**
   - Labels are algorithmic and come from one session, so get more sessions before claiming generalisation.
   - Five of the 453 tracks contain hot-pixel rows. Confirm with the dataset authors that these pixels are defects.

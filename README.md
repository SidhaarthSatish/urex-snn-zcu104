# UREx: event-driven satellite-streak detection with a spiking neural network on the AMD ZCU104

Code, hardware design and results for the NUS Undergraduate Research Experience (UREx) project
**"Resident Space Object Detection and Classification Using Spiking Neural Networks on Event-Based Data"**
(Sidhaarth Satish, supervised by Professor Ankit Srivastava, Department of Electrical Engineering, NUS).

An event camera (Sony IMX636, 1280 x 720) watched the night sky. A small fully connected spiking neural
network (FC-SNN) decides, for each camera event, whether it belongs to a satellite streak. The network is
quantised to INT8 and implemented as a hand-written Verilog core for the ZCU104 (Zynq UltraScale+ XCZU7EV).

> **Status: not run on a board.** The ZCU104 was not available during the project. Every latency and power
> figure below is a simulation or Vivado-tool estimate, not a measurement. The bitstream in this repo has
> **never been loaded on hardware**.

## Results (test set: 19,497 samples, 4,988 streak)

| Model | F1 | AP | Notes |
|---|---|---|---|
| Count threshold | 0.899 | 0.879 | simple baseline |
| Gradient-boosted trees | 0.934 | 0.968 | simple baseline |
| CSNN | 0.943 | 0.987 | 1 seed; dropped |
| **FC-SNN (main)** | **0.953** | 0.989 | 3 seeds; 65,922 parameters |
| Stacked CNN (backup) | 0.958 | 0.992 | 3 seeds; 270,066 parameters |

- INT8 FC-SNN F1 is 0.9531 / 0.9527 / 0.9525, against 0.9529 / 0.9532 / 0.9523 in float.
- RTL simulation matches the Python integer model bit-for-bit on 1,511 of 1,511 test patches.
- Estimated work per decision: about 1,946 synaptic operations (FC-SNN) against 2.18 M MACs (CNN).
- Labels in the dataset are algorithmic and come from a single observation session. Generalisation is not shown.

## Hardware results (Vivado 2018.2, xczu7ev-ffvc1156-2-e, full chip, 16 lanes, 150 MHz)

| | |
|---|---|
| CLB LUTs / registers | 9,569 (4.15 %) / 4,614 (1.00 %) |
| DSP48E2 / RAMB36 | 16 / 19 |
| Timing | met: WNS +1.812 ns, WHS +0.012 ns (hold slack is thin) |
| Vectorless power | 3.572 W (2.879 W dynamic, 0.693 W static), confidence Medium, mostly the Zynq PS |
| Core alone, out of context, 200 MHz | 8,413 LUTs, WNS +0.693 ns |
| Cycles per patch (simulation, mean) | 334, i.e. 1.67 us at 200 MHz |

## Repository layout

```
zcu104_framework/        main project (read its README.md for the full pipeline description)
  rtl/                   snn_core.v, snn_axi.v, urex_top.v (Verilog)
  sim/                   testbench and golden test vectors
  vivado/                build scripts (build2.tcl, build.tcl, dbg_synth.tcl, synth_ooc.tcl)
  vivado/out2/           the built bitstream and its reports (see below)
  python/                reference pipeline, integer SNN model, training, export, checks
  host/                  C pipeline, PYNQ driver, benchmark script (board side, untested on a board)
  weights/               INT8 SNN weights, AXI write list, backup CNN (PyTorch / ONNX)
  config/  data/         pipeline settings; derived data (see "Data" below)
  run_all_tests.sh       runs every offline check
ml_work/                 scripts and result files behind the model comparison (F1/AP, op counts)
```

## The bitstream

`zcu104_framework/vivado/out2/`

| File | What it is |
|---|---|
| `urex_snn.bit` | bitstream for the ZCU104 (md5 `d5f002422e0302f7d4aec14ecdf7ab55`) |
| `urex_snn.hwh` | hardware hand-off file, needed with the `.bit` by PYNQ |
| `urex_snn.hdf` | hardware description (Vivado 2018.x format; newer tools use `.xsa`) |
| `*.rpt` | utilisation, timing and power reports for this exact build |

Built on 14 Oct 2026 with `build2.tcl` in Vivado 2018.2 (Windows), after the last change to the Verilog.
It has **not** been loaded on a board.

## Rebuilding in a newer Vivado (2019.2 or later)

1. Install Vivado with Zynq UltraScale+ MPSoC device support (XCZU7EV is on AMD's free Standard-edition device list;
   confirm in your installer) and the **ZCU104 board files**.
2. From `zcu104_framework/vivado`:
   `vivado -mode batch -source build2.tcl -tclargs 16 150`
   `build2.tcl` is the version that worked. `build.tcl` is meant for newer Vivado but has **never been run**
   and may need small fixes.
3. Check the log for `Using board part ...`. If it warns that no ZCU104 board part is installed, the processor
   settings are generic and the bitstream is not suitable for a real board.
4. Run `dbg_synth.tcl` and check that `mem_reg` has 2,560 flip-flops, and search the synthesis log for
   `Synth 8-3332`. This catches a silent synthesis error where Vivado 2018.2 removed 15 of 16 neuron lanes
   when the RTL was written as a shared-temporary `for` loop. Simulation cannot detect it.
5. A rebuild writes to `vivado/out2/` and overwrites the committed files. New numbers will differ slightly from the
   2018.2 results above. Label them with the Vivado version.

## Running the offline checks

`cd zcu104_framework && ./run_all_tests.sh` (Python with numpy and PyTorch; iverilog for the RTL simulation).
The testbench `sim/tb_snn_axi.v` can also be run in Vivado's xsim; the vector paths may need adjusting.

## Bring-up on a ZCU104 (not yet done)

See "Bring-up on the ZCU104" in `zcu104_framework/README.md`: boot a PYNQ image, copy the folder, load
`urex_snn.bit` with its `.hwh`, then run `host/bench.py`. The first hardware check is `bit_exact.mismatches == 0`.

## Not done

Board run and any latency or power measurement; 200 MHz full-chip rebuild; Vitis AI / DPU path for the CNN backup;
evaluation on more observation sessions.

## Data and ownership

The SBSS (Space Based Surveillance System) event-sensor dataset was created by Mr Ethan Foo at the Satellite
Technology and Research (STAR) Centre, NUS, and is unpublished. `zcu104_framework/data/` contains small
derived samples (patches, a 3.9 M-event raw window) used by the tests. **Keep this repository private** unless the
dataset owner agrees otherwise. No licence has been chosen.

## Reference

Earlier STAR Centre work: M. Y. Chantiswara (2026), UROP report on FPGA-based SNNs for event-driven RSO detection on
the Kria KV260.

"""Offline check of snn_driver.py: a fake MMIO records writes; weight writes must equal the exported AXI write list,
and classify() must drive the register protocol the RTL testbench uses (FIFO clear, pushes, start, poll)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from snn_driver import SnnAccel
from urex import SoftSNN
import numpy as np

class FakeMMIO:
    def __init__(self, sw): self.w = []; self.ev = []; self.sw = sw; self.res = (0, 0)
    def write(self, a, d):
        self.w.append((a, d))
        if a == 0x0 and d & 2: self.ev = []
        if a == 0xC: self.ev.append(d)
        if a == 0x0 and d & 1: self.res = self.sw.classify(np.array(self.ev, np.uint16))
    def read(self, a):
        return {0x20: 0x534E4E01, 0x24: (16 << 24) | (10 << 16) | 128, 0x0: 0x2,
                0x10: self.res[0] & 0xFFFFFFFF, 0x14: self.res[1], 0x1C: 0}[a]

wj = os.path.join(ROOT, "weights/snn_int8.json")
fm = FakeMMIO(SoftSNN(wj)); hw = SnnAccel(mmio=fm); hw.load_weights(wj)
exp = [tuple(int(v, 16) for v in l.split()) for l in open(os.path.join(ROOT, "weights/snn_axi_writes.txt"))]
assert fm.w == exp, "weight write sequence differs from export"
print("weight writes identical:", len(exp))
acc, d, c = hw.classify([0x0001, 0x0203, 0x1405]); print("classify via fake MMIO:", acc, d)

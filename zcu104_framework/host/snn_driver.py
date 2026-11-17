"""PYNQ driver for snn_axi on the ZCU104.

    from snn_driver import SnnAccel
    acc = SnnAccel("urex_snn.bit")              # loads the overlay (urex_snn.hwh must sit next to the .bit)
    acc.load_weights("weights/snn_int8.json")
    score, decision, cycles = acc.classify(ev_words)   # ev_words: sorted uint16 (bin << 9 | addr)

Register map: see rtl/snn_axi.v. NOT YET RUN ON A BOARD.
"""
import json
import numpy as np

REG_CTRL, REG_STATUS, REG_EVENT, REG_ACC, REG_DEC, REG_ACC_TH, REG_CYCLES, REG_ID, REG_PARAMS = \
    0x000, 0x004, 0x00C, 0x010, 0x014, 0x018, 0x01C, 0x020, 0x024
W1_BASE, B_BASE, TH_BASE, WD_BASE = 0x10000, 0x20000, 0x20400, 0x20800
FIFO_DEPTH = 8192


class SnnAccel:
    def __init__(self, bitfile=None, base_addr=0xA0000000, mmio=None):
        if mmio is None:
            from pynq import Overlay, MMIO
            if bitfile:
                self.overlay = Overlay(bitfile)
                ip = [k for k in self.overlay.ip_dict if "snn" in k][0]
                base_addr = self.overlay.ip_dict[ip]["phys_addr"]
            mmio = MMIO(base_addr, 0x40000)
        self.m = mmio
        ident = self.m.read(REG_ID)
        if ident != 0x534E4E01:
            raise RuntimeError(f"unexpected ID register {ident:#x}; is the overlay loaded?")
        prm = self.m.read(REG_PARAMS)
        self.lanes, self.T, self.NH = prm >> 24, (prm >> 16) & 0xFF, prm & 0xFFFF

    def load_weights(self, path):
        w = json.load(open(path))
        W = np.array(w["W"], np.int8).view(np.uint8).astype(np.uint32)       # (512, 128)
        words = (W[:, 0::4] | (W[:, 1::4] << 8) | (W[:, 2::4] << 16) | (W[:, 3::4] << 24)).reshape(-1)
        for i, d in enumerate(words):
            self.m.write(W1_BASE + 4 * i, int(d))
        for j in range(self.NH):
            self.m.write(B_BASE + 4 * j, int(w["B"][j]) & 0xFFFF)
            self.m.write(TH_BASE + 4 * j, int(w["TH"][j]) & 0xFFFF)
            self.m.write(WD_BASE + 4 * j, int(w["WD"][j]) & 0xFF)
        self.m.write(REG_ACC_TH, int(w["ACC_TH"]) & 0xFFFFFFFF)
        self.acc_th = int(w["ACC_TH"])

    def classify(self, ev_words, timeout_polls=1_000_000):
        ev = np.asarray(ev_words, np.uint16)
        if len(ev) > FIFO_DEPTH:
            raise ValueError("patch has more events than the FIFO holds")
        m = self.m
        m.write(REG_CTRL, 0x2)                                  # clear FIFO
        for e in ev:
            m.write(REG_EVENT, int(e))
        m.write(REG_CTRL, 0x1)                                  # start
        for _ in range(timeout_polls):
            if (m.read(REG_CTRL) & 0x3) == 0x2:
                break
        else:
            raise TimeoutError("SNN core did not finish")
        acc = m.read(REG_ACC)
        acc = acc - (1 << 32) if acc & 0x80000000 else acc
        return acc, m.read(REG_DEC) & 1, m.read(REG_CYCLES)

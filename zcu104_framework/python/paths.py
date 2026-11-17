"""Where the framework finds its data. Override with environment variables if your layout differs.
  UREX_DATA   folder with p16.npz, raw_window_b64.npz and models/ (trained .pt + res_*.json)  [default: <framework>/data]
  UREX_MASKED path to masked_events.npz (2.08 M events after hot mask)  [default: ../ml_work/cache/, else data/]
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("UREX_DATA", os.path.join(ROOT, "data"))
MODELS = os.path.join(DATA, "models")
P16 = os.path.join(DATA, "p16.npz")
RAW_WINDOW = os.path.join(DATA, "raw_window_b64.npz")
_cands = [os.environ.get("UREX_MASKED", ""), os.path.join(ROOT, "..", "ml_work", "cache", "masked_events.npz"),
          os.path.join(DATA, "masked_events.npz")]
MASKED = next((c for c in _cands if c and os.path.exists(c)), _cands[-1])

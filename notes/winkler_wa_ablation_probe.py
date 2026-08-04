#!/usr/bin/env python
"""One-shot ∂J/∂φ analytic vs FD under COIL_FEM_VJP_ABLATION / drop-w_a.

Usage:
  python notes/winkler_wa_ablation_probe.py
  python notes/winkler_wa_ablation_probe.py --ablation freeze_k
  python notes/winkler_wa_ablation_probe.py --ablation freeze_sdofs_geom
  python notes/winkler_wa_ablation_probe.py --drop-wa
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow `python notes/...` without install edge cases.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

# RTX 4060-class cards OOM on CUDA graphs during objective FD; disable graphs
# and leave headroom when this probe is spawned under a parent pytest process.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tests.test_winkler_wa_vjp import _phi_grad_once, _print_row

_VJP_ABLATION_ENV = "COIL_FEM_VJP_ABLATION"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation",
        choices=("none", "freeze_k", "freeze_sdofs_geom"),
        default="none",
    )
    p.add_argument("--drop-wa", action="store_true")
    args = p.parse_args()
    os.environ.pop(_VJP_ABLATION_ENV, None)
    d_an, d_fd, J0 = _phi_grad_once(ablation=args.ablation, drop_wa=args.drop_wa)
    label = args.ablation + ("+drop_wa" if args.drop_wa else "")
    print(f"label={label}", flush=True)
    print(f"J0={J0:.16e}", flush=True)
    _print_row(label, d_an, d_fd)
    scale = max(abs(d_fd), abs(d_an), 1.0)
    if scale > 0:
        print(f"dJh/FD={d_an / d_fd:.8f}  rel_err={abs(d_an - d_fd) / scale:.6e}", flush=True)


if __name__ == "__main__":
    main()

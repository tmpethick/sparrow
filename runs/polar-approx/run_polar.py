"""experiments/nanogpt.py with the polar factor of every direction (sparrow.approx_oracles._polar) computed by
PolarExpress (sparrow_jax.polar_factor, float32, --polar_steps iterations) instead of the exact SVD.

    python runs/polar-approx/run_polar.py --polar polarexpress --polar_steps 5 <nanogpt.py args>

--polar svd leaves the library untouched. The patch covers Muon's spectral, Sparrow's RowClip Spectral Row and
both polar factors of Aurora; the eval-time norms (sigma_max) still use the exact SVD.
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]

import sparrow.approx_oracles as ao  # noqa: E402
import sparrow_jax  # noqa: E402
import nanogpt  # noqa: E402


def pop(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


polar, steps = pop("--polar", "svd"), int(pop("--polar_steps", "5"))
assert polar in sparrow_jax.POLAR_METHODS, polar
if polar != "svd":
    ao._polar = lambda m: sparrow_jax.polar_factor(m, polar, steps)
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
json.dump(dict(polar=polar, polar_steps=steps), open(os.path.join(out, "polar.json"), "w"))
nanogpt.main()

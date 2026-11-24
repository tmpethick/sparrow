"""Sparrow: the joint row/spectral constraint set and its oracles.

    sparrow.approx_oracles      directions of the joint row/spectral set used in training: Muon,
                                RowClip∘Spectral, Sparrow, Aurora, the exact LMO by DRS (DenseGeometry:
                                LMO and projection for projected training, JAX)
    sparrow.iterative_solvers   accurate iterative solvers of the joint LMO for benchmarking (NumPy)

The stand-alone Sparrow references `sparrow_jax.py` and `sparrow_torch.py` sit next to this
package and do not depend on it.
"""

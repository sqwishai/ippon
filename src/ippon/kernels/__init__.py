"""Fused Metal kernels (`mx.fast.metal_kernel`, compiled at first use), each with the same numbers as the MLX
composition it replaces in `arch.ops`: one read and one write where MLX would make several passes.

GPU source lives in metal/; Python selects variants, supplies template parameters and launches the kernels.
"""

# Metal source

This folder holds the GPU arithmetic, and the sibling Python modules handle shapes, dtypes, variant selection,
template parameters and launches. `_source.metal_source` reads these UTF-8 package resources once, from any working
directory. Both the wheel and the source distribution include them.

These files are inputs to `mx.fast.metal_kernel`, not standalone translation units. MLX supplies the kernel
signature, its prelude, the input and output buffers and the template constants. Python inserts the few source
fragments and scalar literals that vary between compiled kernels, and it generates no arithmetic.

| Files | Role |
| --- | --- |
| `gemm`, `gemm_thin`, `gemm_glu`, `gemm_tile_loop` | Dense products and their shared tile loops |
| `gemm_simd`, `gemm_glu_simd` | Dense products and the fused GLU on the SIMD-group tiles of M1 to M4 |
| `qgemm`, `qgemm_staged` | Quantized product body and shared double-buffered loop |
| `qgemm_affine`, `qgemm_fp`, `qgemm_*_loader` | Separate affine and floating-point quantization loaders |
| `norm*`, `conv*`, `delta`, `glu` | Normalization, convolution, DeltaNet preparation and gated activations |
| `activation_*`, `erf` | Shared activation statements and the FP32 error-function helper |

All names above have the `.metal` extension. Each Python wrapper replaces its body placeholders, such as
`ACTIVATION`, `LOADER`, `EPS`, `SCALE` and `MASK_ARGS`. MLX's template argument list supplies the other uppercase
names, such as `T`, `BM`, `WIDTH` and `CAUSAL`. Fragment readers remove their final newline where it is not part of
the assembled program. MLX caches kernels by their assembled source, so change whitespace only on purpose.

`gemm.inline` resolves MLX headers from the installed MLX wheel and removes duplicate includes. Shared kernel
helpers adapted from MLX keep their attribution in the wrapper docstrings and the project's `NOTICE`.

For a source-only move, compare the complete `source`, `header` and argument metadata passed to
`mx.fast.metal_kernel` before and after the move. The extraction kept all 149 captured variants byte for
byte on the M5 Pro with MLX 0.32.4, including the tuning header hash. A change to the arithmetic or the tiles needs
the correctness and performance checks in `CONTRIBUTING.md`. If stored measurements would change meaning, it also
needs a `choices.RULE` bump. GPU regression tests cover the compositions in `tests/test_kernels.py`, `test_gemm.py`,
`test_qgemm.py` and `test_attention.py`.

[kernels.md](../../../../docs/kernels.md#every-metal-source) maps every source to its launcher and callers.

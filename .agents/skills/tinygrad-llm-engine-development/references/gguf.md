# GGUF layouts and independent oracles

Read `tinygrad/llm/gguf.py`, `Transformer.from_gguf` in `model.py`,
`Linear.set_quantized` in `kernels/amd.py`, and the selected assertions in
`test/unit/test_gguf.py`. Format support is not model-architecture support.

## Container and model mapping

The parser accepts GGUF versions 2 and 3 and uses little-endian readers; do not
infer full big-endian support from the v3 format specification. Descriptors give
offsets relative to the aligned tensor-data section, whose default alignment is
32 bytes. GGUF dimensions reverse into tinygrad's logical shape. Validate with
non-square tensors so an incorrect transpose cannot hide.

Path input can merge numbered splits; it must start with `split.no=0` and a
`*-00001-of-NNNNN.gguf` path. Tensor input cannot resolve sibling files. Metadata
comes from the first split; later tensor dictionaries are merged. Do not claim
cross-split consistency or duplicate-name rejection without new tests.

`from_gguf` separately maps architecture metadata, tied output embeddings,
optional Q/K norm and bias, MLA ranks, dense/MoE/hybrid blocks, and rotary row
permutations. `HALF` defaults to casting state to float16; `REALIZE` changes eager
materialization versus on-the-fly unpacking. A passing packed-block decoder test
does not validate these transformations or cached model outputs.

## Packed layout checklist

Use `_GGML_NATIVE` and `_GGML_QUANT` as the current dispatch inventory, and primary
GGML layout/dequantization references in `sources.md` for an independent oracle.
Never derive support from the CLI's model-alias comments. Reviewed families:

- Native F32/F16, signed integers, F64 and BF16 use byte slicing/bitcasts.
- Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 have 32 elements per block, different signed offsets,
  minima and high-bit planes. Q8_0 is 2-byte fp16 scale plus 32 int8 values.
- Q2_K through Q6_K have 256 elements per block. Preserve subblock ordering,
  signed scale fields, high masks and minima; nominal bit width is insufficient.
  Q4_K/Q5_K/Q6_K use 144/176/210 bytes, respectively.
- IQ formats combine lookup grids, signs and packed scales. IQ1 needs signed grid
  conversion and delta offsets; IQ4_NL's nonlinear values are not linear int4.
- MXFP4 is 17 bytes for 32 elements, including exponent-byte 0/1 edge behavior.
  Q1_0 is type 41, 18 bytes for 128 elements, with its own bit order. The reviewed
  gguf-py 0.18 oracle lacks that enum, so its existing test uses an explicit ID.

Bound fixtures to whole quantization blocks, finite scale fields and a few rows.
Check exact decoded values/shape/order where appropriate; use justified dtype-
aware tolerances for matrix products and model outputs. Do not require a lossy
quantize/dequantize round trip to recover original floats. Random packed bytes
can encode NaN/Inf scales: distinguish deliberate nonfinite tests from ordinary
finite arithmetic cases. Direct decoder output shape varies; compare flattened
values or the container's final logical reshape deliberately.

`Linear.set_quantized` only recovers original packed weights from the exact
expected dequantization expression under storage/order-preserving views. It must
reject subsequent arithmetic or row permutations, including RoPE permutations.
AMD Q6 padding to 212-byte word-aligned blocks is an internal kernel layout,
**not a change to the 210-byte GGUF format**. Generic decoder support exceeds the
custom Q4_K/Q5_K/Q6_K/IQ4_XS kernel subset.

## Existing tests: what the oracle proves

- `TestGGUF.test_dequantization_*_hardcoded`: hand-calculated blocks. Q3_K checks
  both high-mask polarities. IQ4_NL shares the production table, so this case
  alone verifies packing/scaling, not independent table correctness.
- `TestGGUF._test_dequantization`: four blocks against external `gguf.dequantize`.
  Quantizer-backed input is not uniformly seeded; the unsupported-quantizer byte
  fallback uses seed 42. Record/pin the oracle package version for new golden data.
- `TestGGUFTables`: IQ grids versus external gguf-py tables. A second library is
  useful differential evidence, not an infallible specification.
- `test_dequantization_mxfp4_old`: scalar sign/exponent/mantissa oracle;
  `test_dequantization_mxfp4_block`: embedded bytes and frozen expected values,
  with provenance comments, **no runtime download**.
- `test_multi_part_load`: tiny temporary F32 splits and a missing-part error.
  `test_expected_failure_unknown_type`: rejects an unknown type.
- `TestGGUFGEMV`: external dequantization and NumPy product reference, finite
  scale repair, but ordinary shapes are 8192x2048; offline does not mean cheap.
- `test_load_*` and `TestGGUFGC` fetch GGUF files. Do not run them by default.

## Synthetic fixture design (recommended additions, not existing coverage)

Start with `TestGGUF._build_gguf` as a **limited example**, not a general writer:
it fixes v3/alignment=32, supports only string/UINT32 metadata, uses character
rather than UTF-8 byte length for string values, and concatenates payloads without
per-tensor padding. Its current ASCII, one-tensor-per-split fixtures avoid those
limitations. For richer fixtures use a reviewed test-side writer or extend the
helper with byte lengths and aligned offsets; no production abstraction is needed.

Test metadata arrays/UTF-8, custom alignment, non-square/multiple tensors, v2/v3,
invalid magic/version/type, truncation and split naming/start rules. Some malformed
inputs currently fail incidentally rather than with a designed validation error;
state the expected contract before changing it. Quantized divisibility, duplicate
names and inconsistent split metadata are not comprehensively validated today.

For a new format combine an independent scalar golden block (nonzero minima,
signs/high bits and extreme finite scales) with external decoder comparison and
a tiny dequantized matrix product. Check the same bytes through parsing and
weight mapping before using any already-local real model. Use the existing
property-testing skill for bounded bit patterns and metadata cases, with its
floating-point and resource caveats.

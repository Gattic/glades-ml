## Columnar / memory-mapped datasets (`.gcol`)

This directory contains a minimal implementation of **memory-mapped float32 matrices** for production-scale training/inference.

### Why

`NumberInput` uses `shmea::GMatrix`, which is convenient but requires loading the full dataset into RAM. For large datasets, that becomes the bottleneck.

The engine’s training loop already supports **zero-copy row access** via `DataInput::get*RowView()`. The `.gcol` path plugs into that contract using `mmap(2)` so rows can be accessed without allocations or full-dataset loads.

### File format

`.gcol` is a simple binary container for a dense **row-major** float32 matrix.

- **Header size**: 64 bytes (fixed)
- **Endianness**: little-endian integers
- **Payload**: `rows * cols` float32 values

See `MappedMatrix.h` for the exact header layout.

### On-disk dataset layout

`MappedNumberInput` expects a directory containing:

```
train.x.gcol   # features (float32)
train.y.gcol   # expected outputs (float32)
test.x.gcol    # optional
test.y.gcol    # optional
```

### Exporting from `NumberInput`

If you already have a `NumberInput` instance with populated `trainMatrix/trainExpectedMatrix` (and optionally `testMatrix/testExpectedMatrix`), you can export it:

- Call `NumberInput::exportMappedDataset("<dir>")`

This writes `train.x.gcol`, `train.y.gcol`, and (if present) `test.*.gcol`.

### Loading with `MappedNumberInput`

- Construct `MappedNumberInput`
- Call `import("<dir>")`

Then training/inference will read rows directly from the memory-mapped files via `getTrainRowView()` / `getTestRowView()`.

### Notes / limitations

- `.gcol` is **dense float32 only** (no categorical strings, no sparse column blocks).
- This is intended for datasets **after preprocessing/encoding**.
- `MappedNumberInput` does **not** fit preprocessing; it assumes you already produced numeric tensors.


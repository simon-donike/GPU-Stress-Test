# Measurement and FLOP accounting

GPU Stress Test reports an analytical dense-GEMM rate. It is useful for comparing runs made with the same model, precision, device selection, and software stack. It does not claim to measure every instruction executed and is not a substitute for a hardware profiler.

## FLOP convention

One fused multiply-add counts as two floating-point operations. For a linear layer with input width `in` and output width `out`, a forward matrix multiply contributes `2 × B × in × out` FLOPs for a batch of `B` samples.

For training, the first linear layer contributes `4 × B × I × H`: the forward pass and weight-gradient path are counted, while the input-gradient path is omitted because it is not needed by an earlier layer. Every later hidden or output linear layer contributes `6 × B × in × out`, covering forward, input-gradient, and weight-gradient matrix multiplications. Bias addition, activations, MSE loss, Adam updates, random-number generation, tensor copies, host-to-device transfers, and distributed communication are excluded from the FLOP numerator.

The optional independent square workload adds `2 × R × D³` FLOPs per batch, where `R` is `--extra-matmul-repeats` and `D` is `--extra-matmul-dim`. These operations are forward-only and independent of the training batch, so this term is not multiplied by `B`.

The implementation keeps forward, training, and extra-work terms separate in the CSV. `total_flops` is the training term plus the extra term for that rank's row. The analytical parameter-memory lower bound is 16 bytes per parameter: FP32 parameters, FP32 gradients, and two FP32 Adam moment buffers. It excludes activations, allocator fragmentation, temporary GEMM workspace, and the reusable extra-matmul operand.

`training_flops` already includes the forward pass; do not add `forward_flops`
to it. These are integer counts under the dense `2 × M × N × K` convention,
not hardware performance-counter readings or exact scalar instruction counts.
The estimator assumes every linear weight is trainable and the input tensor
does not require a gradient. The runner checks that the instantiated model
agrees with the dependency-free estimator before training.

## Timed windows

Initial `--warmup-steps` batches are excluded. Each subsequent window starts after a device synchronization at batch start and ends after the final batch synchronization. The denominator includes Python, loader gaps, optimizer work, and the finite-loss check/all-reduce inside each batch. Telemetry queries, CSV logging, and distributed boundary gathering occur after the timed endpoint and are excluded. The first batch's loader and transfer work occurs before the timing callback and is excluded from the window. A final partial window is saved at epoch end.

For distributed windows, each rank reports its own FLOP count and elapsed time. Global throughput is the sum of rank FLOPs divided by the maximum rank elapsed time, reflecting the slowest rank. The repeated `tflops_total` and `samples_per_second_total` fields contain those global window rates; rank rows retain rank-local `total_flops` and `tflops_per_device`. Aggregate summaries use one global FLOP total per window and divide the sum of those totals by the sum of global window durations; they are weighted by measured time rather than averaging per-window rates.

Rows are identified by `window`, `rank`, and `device_uuid`. Telemetry is matched to the current CUDA device UUID. A missing command, unreadable field, or UUID mismatch leaves that value blank; it is never converted to zero. The sampled thermal option checks before the first workload, during warmup, and after measured windows, and stops at or above the configured threshold or when telemetry is unavailable. Sampling cannot provide a hard real-time safety guarantee.

## CSV schema

| Fields | Meaning |
| --- | --- |
| `timestamp`, `window`, `step`, `epoch` | UTC save time and training position |
| `rank`, `local_rank`, `world_size` | Distributed identity |
| `device_type`, `device_name`, `device_uuid` | Device identity; UUID is the telemetry join key |
| `batch_count`, `sample_count` | Work represented by the row |
| `elapsed_seconds`, `window_seconds` | Per-rank elapsed time and distributed denominator |
| `forward_flops`, `training_flops`, `extra_flops`, `total_flops` | Analytical numerator components |
| `tflops_per_device`, `tflops_total`, `samples_per_second_total` | Rank-local and repeated global derived rates |
| `gpu_temp_c`, `gpu_util_pct`, `gpu_power_w`, `gpu_power_limit_w` | Sampled board telemetry |
| `gpu_mem_used_gib`, `gpu_mem_total_gib`, `gpu_mem_used_pct` | Board-reported memory, in GiB |
| `cuda_peak_allocated_gib`, `cuda_peak_reserved_gib` | PyTorch allocator peaks for the measured window, in GiB |

Board memory and allocator memory answer different questions. Board totals include driver and other-process use; allocator peaks describe PyTorch's view. They should not be compared as the same counter.

`step` is Lightning's optimizer-step counter, while `batch_count` and
`sample_count` describe the actual work in that window. FP16 gradient scaling
can skip an optimizer update on overflow even when matrix multiplications
ran; these operation counts still describe that executed forward/backward
work. No optimizer FLOPs are included. A finite loss check is an early failure
signal, not a numerical correctness or GPU memory-integrity test.

## Interpreting reports

Use `summary.json` for the saved aggregate and `summary.png` or `report.pdf` for trends. A run's metadata records status (`running`, `completed`, `failed`, or `interrupted`); an incomplete run should be treated as partial evidence. Compare runs only after checking saved configuration and precision metadata. Legacy CSVs may be read for convenience, but inferred FLOPs are explicitly labeled `legacy/unverified`; they are not certified measurements. The [README illustration](../README.md#illustrative-report) uses invented data with documented arithmetic; it is not a measured result or rendered report.

## Reference documentation

The implementation follows the [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html), [PyTorch mixed-precision guidance](https://pytorch.org/docs/stable/amp.html), and the [NVIDIA `nvidia-smi` query documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html). These links describe the underlying device synchronization, precision, and telemetry interfaces; the FLOP convention above is the application's explicit accounting policy.

# GPU Stress Test

GPU Stress Test runs a configurable synthetic dense network and records windowed throughput together with GPU telemetry. It is intended for repeatable relative comparisons, cooling checks, and capacity checks. Reported TFLOP/s is an analytical estimate for dense matrix multiplications in the timed host window, not a hardware benchmark or vendor peak specification.

## Requirements

- Python 3.10 or newer
- PyTorch 2.5 or newer and a compatible CUDA driver for GPU mode
- An NVIDIA driver and `nvidia-smi` for temperature, utilization, memory, and power telemetry

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU mode is the default and fails clearly when CUDA is unavailable; it never silently changes to CPU mode. Request CPU mode explicitly with `--accelerator cpu --precision 32-true`.

## Safe first steps

Validate a configuration without importing PyTorch, touching a GPU, creating an output directory, or writing files:

```bash
python run.py --dry-run
```

The defaults are deliberately moderate, but no setting can guarantee a fit on every card. Start small, then increase dimensions while watching memory and temperature:

```bash
python run.py --batch-size 16 --input-dim 512 --hidden-dim 1024 \
  --output-dim 512 --num-hidden-layers 2 --num-batches 20 --num-epochs 1
```

Run all visible GPUs with `--devices -1`; use `--devices 1` for one GPU. `--generate-on-device` (the default) avoids CPU data-loader cost. Use `--no-generate-on-device` when host-side generation and transfer are part of the workload. `--max-temperature-c` is a sampled stop guard and requires matched readable telemetry; it is not a real-time thermal safety system.

Each run gets a unique directory below `metrics/` by default. Existing run folders are never reused. A completed run contains the CSV, `config.json`, `report.pdf`, `summary.png`, and `summary.json` when reporting is enabled. Disable reporting with `--no-report`.

## Useful options

`--batch-size`, `--num-batches`, `--num-epochs`, `--input-dim`, `--hidden-dim`, `--output-dim`, and `--num-hidden-layers` define the synthetic model and schedule. `--extra-matmul-repeats` and `--extra-matmul-dim` add independent forward-only square GEMMs; increase them cautiously because memory use grows with the square operand.

`--warmup-steps` excludes initial batches from throughput windows. `--log-every-n-steps` controls the normal window size; a final partial window is retained. `--precision auto` selects BF16 mixed precision when every selected GPU has compute capability at least 8.0, otherwise FP16 mixed precision. The effective precision is saved in run metadata. `32-true` uses FP32 tensors; the `--float32-matmul-precision` setting controls whether internal FP32 matmul arithmetic uses the highest, high, or medium policy.

See all defaults and validation rules with `python run.py --help`.

## Batch autotuning

Use `--autotune` to calibrate batch size for the exact workload configuration before a later ordinary run. It starts at half the requested batch, doubles eligible candidates, and backs off after OOM or allocator-limit failures. The default limits are eight trials, eight measured seconds per candidate, a 120-second child timeout, a 95% utilization target, and an 85% CUDA allocator fraction:

```bash
python run.py --autotune --batch-size 512 --tune-max-batch-size 4096 \
  --tune-max-trials 8 --tune-seconds 8 --tune-timeout-seconds 120 \
  --tune-target-utilization 95 --tune-memory-fraction 0.85
```

Calibration is GPU-only, Linux-only, rejects MIG and busy GPUs, and runs each candidate in a fresh isolated process. Missing or mismatched telemetry aborts it. The selector first keeps candidates within 3% of the global best weighted throughput, then prefers target utilization on every GPU and the smallest batch within that shortlist. It writes trial CSV/JSON results, an `autotune.png` plot, a recommended config, and a command; it never starts the full run automatically. See [docs/autotuning.md](docs/autotuning.md) for the safety model and selection details.

| Setting | Default |
| --- | --- |
| Batch size / batches per epoch / epochs | 64 / 100 / 1, per device |
| Input / hidden / output width | 1024 / 2048 / 1024 |
| Hidden layers | 4, plus the output layer |
| Warmup / measured batches per window | 5 / 10 |
| Devices / precision | 1 / `auto` |
| CPU loader workers | 0 |
| Extra matmul repeats / dimension | 0 / 2048 |
| Seed / Adam learning rate | 42 / 0.001 |

Warmup counts toward the requested batches, so `num_batches × num_epochs` must
exceed `warmup_steps`. Use `--warmup-steps 0` when intentionally measuring every
batch, including initialization overhead. Multi-GPU execution supports one node;
under `torchrun`, set `--devices` to match `--nproc-per-node`.

The memory preflight rejects a model whose parameter-state lower bound cannot fit
in a selected GPU's total VRAM. With an explicit allocator cap it also probes the
available capped-memory condition; neither check guarantees a fit because
activations, auxiliary operands, optimizer temporaries, and other processes also
need memory. Non-finite losses abort the run. Completed CSV windows are preserved
after failures; `config.json` records the status once output initialization has
succeeded.

## Reports and offline regeneration

Reports are generated only from saved CSV and JSON files. They plot global throughput, per-rank telemetry, and memory peaks, and mark legacy or incomplete FLOP data as unverified. Regenerate on any machine with the reporting dependencies:

```bash
python -m utils.create_pdf \
  --metrics metrics/<run>/metrics.csv \
  --config metrics/<run>/config.json \
  --output metrics/<run>/report.pdf
```

Files in `resources/` are historical example images and a report for visual reference. They are not current measurements or performance claims.

Reporting requires only Matplotlib and the Python standard library. See
[docs/reporting.md](docs/reporting.md) for integrity handling and artifact details.

## Measurement method

The accounting convention and CSV field definitions are documented in [docs/measurement.md](docs/measurement.md). In brief, an FMA counts as two FLOPs. The first linear layer contributes `4 × B × I × H` training FLOPs per batch; every later hidden or output layer contributes `6 × B × in × out`. An extra square GEMM contributes `2 × R × D³` per batch and is not multiplied by batch size. Bias, activation, loss, Adam updates, random-number generation, copies, and communication are excluded from the numerator, while loader gaps and host overhead remain in the timed denominator.

For distributed runs, throughput sums FLOPs across ranks and divides by the maximum elapsed time in the window. Summaries use total FLOPs divided by total window seconds, so long and short windows are weighted correctly. Missing or UUID-mismatched telemetry stays blank; it is never treated as zero. Memory fields distinguish board-reported total/used GiB from PyTorch allocator peaks.

Older results are not directly comparable: the former estimator counted forward
work against training time, extrapolated totals from one rank, and miscounted
extra matmuls. The new extra workload is independent of network predictions,
`--num-hidden-layers 1` now creates exactly one hidden layer, and defaults are
smaller. Legacy reports retain their original rates under an explicit
`legacy/unverified` label; they cannot provide a corrected weighted aggregate.

## Docker

The image uses a pinned PyTorch CUDA runtime base. Build and launch it with the NVIDIA Container Toolkit:

```bash
docker build -t gpu-stress-test .
docker run --rm --gpus all -v "$PWD/metrics:/app/metrics" \
  gpu-stress-test python run.py --devices -1
```

The image command is `python run.py --help`, so starting a container does not start a stress workload. Pass an explicit `python run.py ...` command as above. Keep dimensions and batch size appropriate for the selected GPU.

## Layout

- `run.py`: validated CLI and dry-run entry point
- `utils/config.py`: dependency-free configuration and path validation
- `utils/flops.py`: analytical sizing and FLOP formulas
- `utils/model.py`, `utils/data.py`: synthetic workload
- `utils/monitor.py`: timed windows, distributed aggregation, and telemetry
- `utils/create_pdf.py`: offline PDF, PNG, and JSON report generation
- `utils/runner.py`: PyTorch Lightning orchestration
- `utils/autotune.py`, `utils/tuning_analysis.py`: isolated calibration and evidence-based batch selection
- `utils/tuning_report.py`: offline calibration plot
- `docs/measurement.md`: measurement conventions and schema
- `docs/autotuning.md`: bounded batch-size calibration and safety model

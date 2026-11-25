# Batch-size autotuning

GPU runs calibrate batch size by default. Calibration measures the configured workload at several batch sizes, saves a recommendation, and then the CLI automatically starts the stress test with that configuration. CPU runs skip GPU calibration.

## Choose the run mode

| Command | Behavior |
| --- | --- |
| `python run.py` | Calibrate, then automatically start the GPU stress test. |
| `python run.py --autotune` | Explicitly select the default GPU workflow. |
| `python run.py --no-autotune` | Skip calibration and run the exact requested batch size. |
| `python run.py --autotune-only` | Calibrate and save results; do not start the follow-up stress test. |
| `python run.py --dry-run` | Print the configuration, analytical estimates, and planned mode without running either phase. |

The three mode flags are mutually exclusive. Explicit calibration flags require GPU mode. Keep the workload settings you want to calibrate; the batch size is the requested starting point and must not exceed the tuning ceiling:

```bash
python run.py --autotune --batch-size 512 \
  --tune-max-batch-size 4096 --tune-max-trials 8 \
  --tune-seconds 8 --tune-timeout-seconds 120 \
  --tune-target-utilization 95 --tune-memory-fraction 0.85
```

The defaults are `4096`, `8`, `8` seconds, `120` seconds, `95%`, and `0.85` for those six tuning options respectively. A trial must provide at least three complete telemetry windows to be eligible for selection. Its timed stop waits for both the requested duration and three windows; slow batches may extend measurement beyond the requested seconds, within the whole-process timeout. Normal `--measurement-seconds`, `--min-measurement-windows`, and `--log-every-n-seconds` settings are overridden during calibration. A tighter `--memory-fraction` cap is honored. Calibration uses one-second logging and a high step threshold so time-based windows end the measurement without changing the workload.

The model, dimensions, precision, device selection, data-generation mode, extra GEMMs, and other workload settings are copied exactly. Only `batch_size` changes between candidates. The search begins at half the requested batch size, grows by doubling, and backs off between the largest eligible and smallest memory-failing candidates. OOM and allocator-limit failures drive backoff; thermal, non-finite-loss, timeout, telemetry, or other failures stop calibration instead of being silently retried.

## Safety and scope

Calibration currently requires Linux process groups, GPU mode, physical NVIDIA GPUs, and an ordinary single invocation of `python run.py`. For multi-GPU calibration use `--devices -1` or a GPU count. External `torchrun` launches must pass `--no-autotune` and match `--devices` to their process count. MIG devices are rejected because calibration needs stable physical GPU UUID telemetry. Selected GPUs must be idle and are locked with per-UUID process locks while trials run. Every trial runs in a fresh isolated child process, with its own CUDA context, allocator cap, metrics directory, and failure log. Existing output directories are never reused.

The parent checks device identity, visibility, and idle compute processes before each trial. Each child checks memory headroom before moving the model to a GPU and monitors the sampled temperature guard (default `80°C` when none is supplied). Calibration always applies an allocator cap: the smaller of the requested total-memory fraction and free memory minus reserved headroom. Headroom is at least 1 GiB or the uncapped fraction of total VRAM, whichever is larger. Ordinary runs only apply this additional check when `--memory-fraction` is supplied. Neither check guarantees room for every runtime allocation. The thermal guard stops at or above its threshold. Missing or UUID-mismatched telemetry aborts the calibration; it is not treated as zero. The allocator memory fraction limits PyTorch's caching allocator. It is not a cap on total board memory, and `100%` sampled GPU utilization does not prove compute saturation.

The per-UUID lock prevents concurrent autotuners for the same user and GPU. It does not reserve the GPU from other users or eliminate races with a workload that starts after the idle check; run calibration on machines where the selected GPUs remain dedicated.

Autotuning is a calibration tool, not a thermal or OOM guarantee. The recommendation reflects the current machine, driver, process isolation, model, precision, and workload settings. Recheck it after changing any of those conditions.

## How a recommendation is chosen

An eligible candidate must have complete aligned rank rows, valid timestamps and FLOP counts, at least three complete measured telemetry windows, readable utilization and board-memory telemetry, valid allocator peaks, and no non-finite, thermal, timeout, or memory failure. Throughput is calculated from summed samples over summed global window duration; per-window rates are weighted by elapsed time.

The optimization objective is training samples per second. Estimated TFLOP/s is also reported, but it does not choose the winner. These objectives can differ when extra GEMMs add a fixed amount of work per batch.

The selector first forms a shortlist of eligible candidates whose weighted throughput is within 3% of the global best eligible throughput. It then prefers target-met candidates on every GPU within that shortlist and chooses the smallest batch among them; if none meet the target, it chooses the smallest candidate in the shortlist and marks it below target. This is the best tested candidate under the current bounded search, not a proof of global optimality. No recommendation is produced when the saved evidence is incomplete.

## Output artifacts

The tuning directory contains progressive results so an interrupted search remains inspectable:

- `tuning_results.csv`: one row per candidate, status, weighted rates, utilization, memory fraction, duration, and trial path.
- `tuning_results.json`: complete trial records, requested settings, effective memory and temperature guards, runtime device identities, recommendation, and failure reasons.
- `autotune.png`: offline throughput/utilization/memory plot when reporting is enabled.
- `recommended_config.json`: the full ordinary-run configuration with the selected batch size.
- `recommended_command.txt`: the selected stress-test command, including `--no-autotune` to prevent recalibration, one-second logging, and a high step threshold.
- `trial_*/`: each isolated child run's metrics, metadata, process log, and failure details.

By default, the CLI starts the saved recommendation immediately after calibration and report generation succeed. The stress test uses a fresh sibling directory named in `recommended_config.json`; the tuning directory retains all trial evidence. The follow-up preserves your requested batch/epoch limits, optional measured-duration limit, model, precision, and data mode, while using the selected batch size, calibrated memory/temperature guards, and calibration logging cadence. The calibration timeout applies to calibration children only, not the follow-up stress test.

Calibration errors, missing eligible candidates, or report-generation failures prevent automatic continuation. A valid recommendation below the utilization target still proceeds, with the target result recorded and printed. With `--autotune-only`, inspect the saved command and run it later when ready. The generated command uses a unique run name; choose a fresh `--run-name` if rerunning it after an automatic run has already used that directory.

## Limitations and references

Dependencies must already be installed in the Python environment; autotuning does not install packages. The allocator behavior and memory statistics follow [PyTorch CUDA memory management](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management). Utilization is the sampled NVIDIA device counter described by [`nvidia-smi` query options](https://docs.nvidia.com/deploy/nvidia-smi/index.html); it is a sampled activity indicator, not a proof that every compute pipeline is saturated.

Reported utilization is the duration-weighted mean of sampled activity for each GPU, with the lowest GPU mean used for target assessment. It does not promise every sample reaches the target. The plot distinguishes allocator peaks from sampled whole-board memory usage. Measurements exclude warmup and pauses for telemetry and report writing; they are not whole-process wall-time throughput. Short calibration does not establish sustained thermal equilibrium or long-run stability. If utilization remains below target, inspect model dimensions, host data loading, logging cadence, precision, and multi-GPU communication before increasing memory pressure.

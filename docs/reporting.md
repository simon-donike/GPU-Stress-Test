# Reporting

Reports are generated entirely from saved artifacts. The reporter does not import
PyTorch, initialize CUDA, call nvidia-smi, or infer hardware from the host
where the report is created.

## Inputs and command

The canonical metrics CSV contains one row for every rank and aligned measured
window. Per-rank fields such as total_flops and sample_count are summed across
the ranks in each complete window. Global rates such as tflops_total and
samples_per_second_total are repeated on each rank row and are never summed.

Generate a report later, including on a CPU-only machine, with:

    python -m utils.create_pdf --metrics metrics/run_x/metrics.csv \
      --config metrics/run_x/config.json --output metrics/run_x/report.pdf

The command writes report.pdf, summary.png, and summary.json beside the
requested PDF. Missing telemetry is shown as an empty panel rather than
reconstructed or guessed.

## Canonical CSV fields

The writer should emit:

    timestamp,window,step,epoch,rank,local_rank,world_size,device_type,device_name,device_uuid,batch_count,sample_count,elapsed_seconds,window_seconds,forward_flops,training_flops,extra_flops,total_flops,tflops_per_device,tflops_total,samples_per_second_total,gpu_temp_c,gpu_util_pct,gpu_mem_used_gib,gpu_mem_total_gib,gpu_mem_used_pct,gpu_power_w,gpu_power_limit_w,cuda_peak_allocated_gib,cuda_peak_reserved_gib

forward_flops, training_flops, and extra_flops are counted model work.
They exclude uncounted non-GEMM work. window_seconds covers the measured
training window, including loader gaps inside that window, but excludes
telemetry collection time. Warmup batches must not be emitted as measured
windows.

The aggregate throughput is:

    sum(rank total_flops in each complete window) / sum(window window_seconds) / 1e12

This is time-weighted and uses the slowest rank's elapsed time as the global
denominator. Incomplete or duplicate-rank windows are excluded and surfaced
in summary.json. Per-device throughput is derived from the rank's local count
and elapsed time; it must not be averaged as an unweighted average of rank
rates.

Modern CSVs must contain the count and timing columns. The reporter rejects
missing headers and excludes windows with missing/duplicate/out-of-range ranks,
invalid timestamps, inconsistent work counts, or inconsistent global durations.
`excluded_windows` lists the window IDs and reasons in `summary.json`.
`aggregate_tflops`, `total_flops`, `total_window_seconds`, and `sample_count`
refer only to complete, valid windows. `max_saved_readings` contains sampled
telemetry maxima and saved allocator peaks, not continuous hardware maxima.

The dashboard shows global and per-device throughput, utilization, temperature,
board memory, power, and per-window allocated/reserved memory peaks. Missing
readings remain unavailable. Colors are consistent by rank across panels;
metadata pages preserve saved hardware identities, configuration, and summary
integrity details with wrapping and pagination. All displayed times are UTC.
PDF, PNG, and JSON are staged before replacing existing report artifacts;
input CSV and configuration files are never rewritten.
The three replacements cannot be atomic as a set. `summary.json` is published
last as the completion marker and includes a `report_generation_id` and
`artifact_sha256` hashes for the PDF and PNG. If publication is interrupted,
hash mismatches identify an incomplete artifact set; regenerate from the inputs.

## Compatibility

Older CSV files with tflops_total and tflops_per_gpu are accepted for
reading. They are labeled legacy/unverified in summary.json and in the
dashboard. Legacy values are historical estimates, not verified FLOP
accounting, because their timing and counted-work semantics are not recoverable
from the old schema.

Legacy global rates are displayed once per step and devices are grouped using
their saved GPU IDs. Weighted aggregates, operation counts, and sample totals
remain unavailable because legacy files do not retain their timing denominators.


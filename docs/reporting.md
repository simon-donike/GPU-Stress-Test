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
requested PDF. `summary.png` is a 180-DPI export of the performance and temperature overview.
Missing telemetry is shown as an empty panel rather than reconstructed or guessed.

## Visual standard and page sequence

Every PDF page uses A4 landscape (297 × 210 mm), a navy masthead with a cyan
accent, a light background, consistent margins, a run-status badge, and sequential
page numbers. Embedded DejaVu Sans fonts and vector charts keep text and lines
sharp when zoomed or printed. Numeric units and UTC date formatting are explicit.
Report styling is isolated from local Matplotlib themes; no LaTeX or downloaded
fonts are required.

1. **Performance and temperature overview:** four key-number cards for weighted
   estimated TFLOP/s, peak sampled temperature, sample throughput, and measured
   seconds. Per-GPU temperature curves sit beside throughput, with a configured
   stop line and shared UTC axis limits. The temperature card shows the margin
   to that stop and turns red if a saved sample reaches or exceeds it. Notes show
   accepted/total windows, temperature coverage, and run status.
2. **GPU telemetry:** up to six ranks per page, with a shared rank-color legend
   and aligned UTC time axes. Panels show utilization, temperature, board memory,
   board power, allocator peaks, and estimated device throughput. Configured
   thermal stops, saved memory capacities, and saved power limits appear as
   reference lines only when that information is available. Allocated and reserved
   memory use distinct line styles.
3. **Device comparison:** up to eight devices per page, showing weighted
   TFLOP/s, weighted utilization, sampled temperature/VRAM/power maxima, and
   utilization coverage. Short hardware identities aid scanning; full saved
   identities remain in the appendix.
4. **Configuration and integrity appendix:** wrapped, paginated saved settings,
   hardware metadata, accounting conventions, exclusions, and summary fields.

Rank colors remain consistent across report pages. The overview temperature card,
temperature curves, and telemetry pages include saved readings from windows
excluded from throughput accounting; the device comparison uses accepted windows
only. Missing samples and gaps in modern window IDs interrupt curves. No smoothing, interpolation, or synthetic
readings are added. The overview shows up to six ranks with the highest saved
temperature peaks, so the hottest device remains visible. Telemetry and comparison
pages include all saved ranks. Incomplete/failed run status and
legacy/unverified estimates remain visible instead of being styled as a success.

Temperature is a sampled GPU sensor reading, not a continuous maximum or a
measurement of every board component. Its card uses all readable saved rows,
including rows without usable timestamps; curves require timestamps. Missing
temperatures appear as unavailable/gaps, with readable/saved row counts in the
overview. If no valid stop is saved in the configuration, no stop or margin is
invented. A configured stop remains visible in the plot even if all readings are
below it. The margin is relative to that configured stop, not a hardware rating.

The style lives in `utils/report_style.py`, page layouts in
`utils/report_pages.py`, and parsing, validation, and artifact publication in
`utils/create_pdf.py`.

The [README example](../README.md#illustrative-report) is a hand-authored SVG
illustrating the overview design. Its values are invented and its arithmetic is
documented beside it. It was not produced by the reporter and does not establish
rendering correctness or GPU performance.

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

`per_device_summary` adds weighted per-device throughput and utilization,
coverage counts, hardware identity, and maxima from accepted windows. Local
throughput is `sum(local total_flops) / sum(local elapsed_seconds) / 1e12`.
Utilization uses global window duration as the weight over readable samples;
`utilization_readings / utilization_window_count` makes partial coverage explicit.
Legacy weighted rates remain unavailable. The existing `max_saved_readings`
continues to describe all saved numeric readings, including excluded windows,
so it can differ from accepted-window device maxima. `report_design` identifies
the visual standard. These are additive fields in summary schema version 2.

`temperature_summary` adds `peak_temperature_c`, `configured_stop_c`,
`margin_to_stop_c` (stop minus peak), `readable_rows`, and `saved_rows`, with
`scope: "all_saved_rows"`. Missing peaks, stops, or margins are `null`; a negative
margin means a saved reading exceeded the configured stop. Coverage counts saved
rows, not elapsed-time coverage or unsaved warmup checks. These values also appear
in the integrity appendix. The temperature-focused layout is report design `v2`;
the JSON schema remains version 2 with this additive field.

All displayed times are UTC.
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


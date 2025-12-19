"""
Minimal entrypoint for the GPU stress test.

All heavy lifting lives in utils/. This script simply parses CLI args and
dispatches to the modular runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import StressTestConfig
from utils.runner import run_stress_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dummy PyTorch Lightning model and log GPU metrics.")
    parser.add_argument("--batch-size", type=int, default=StressTestConfig.batch_size, help="Batch size for the dummy run.")
    parser.add_argument("--num-batches", type=int, default=StressTestConfig.num_batches, help="Number of batches to process.")
    parser.add_argument("--num-epochs", type=int, default=StressTestConfig.num_epochs, help="Number of epochs to run.")
    parser.add_argument("--input-dim", type=int, default=StressTestConfig.input_dim, help="Input feature dimension.")
    parser.add_argument("--hidden-dim", type=int, default=StressTestConfig.hidden_dim, help="Hidden layer dimension.")
    parser.add_argument("--output-dim", type=int, default=StressTestConfig.output_dim, help="Output feature dimension.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=StressTestConfig.num_workers,
        help="DataLoader workers. Increase if CPU is the bottleneck.",
    )
    parser.add_argument(
        "--generate-on-device",
        dest="generate_on_device",
        action="store_true",
        default=StressTestConfig.generate_data_on_device,
        help="Generate synthetic data on the GPU to avoid CPU dataloader bottlenecks.",
    )
    parser.add_argument(
        "--no-generate-on-device",
        dest="generate_on_device",
        action="store_false",
        help="Disable on-device data generation.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=StressTestConfig.num_hidden_layers,
        help="Number of hidden linear layers (>=1). Larger values increase parameter count.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=StressTestConfig().metrics_root,
        help="Base directory to store run folders. Default: repo_root/metrics",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run folder. Default: run_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        default=StressTestConfig.log_filename,
        help="Name of the CSV file inside the run folder.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=StressTestConfig.log_every_n_steps,
        help="How often to append metrics to CSV (every N global steps).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=StressTestConfig.devices,
        help="Number of GPUs to use (-1 = all available).",
    )
    parser.add_argument(
        "--extra-matmul-repeats",
        type=int,
        default=StressTestConfig.extra_matmul_repeats,
        help="Number of extra square matmul passes to burn compute each step.",
    )
    parser.add_argument(
        "--extra-matmul-dim",
        type=int,
        default=StressTestConfig.extra_matmul_dim,
        help="Dimension for the extra square matmul (increase to raise FLOPS; watch VRAM).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StressTestConfig(
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_epochs=args.num_epochs,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_workers=args.num_workers,
        generate_data_on_device=args.generate_on_device,
        extra_matmul_repeats=args.extra_matmul_repeats,
        extra_matmul_dim=args.extra_matmul_dim,
        log_every_n_steps=args.log_every_n_steps,
        devices=args.devices,
        metrics_root=args.metrics_dir,
        run_name=args.run_name or StressTestConfig().run_name,
        log_filename=args.log_filename,
    )

    run_stress_test(config)

    print(f"Run complete. Metrics logged to {config.log_path}")
    print(f"PDF report (if generated) at {config.run_dir / 'report.pdf'}")
    print("Sample columns: timestamp, step, tflops, gpu_temp_c, gpu_util_pct")


if __name__ == "__main__":
    main()

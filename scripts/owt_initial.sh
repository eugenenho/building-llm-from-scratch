#!/bin/bash

echo "=== Starting run 1 of 2: gpu-ts-setup-1e2 ==="
uv run python cs336_basics/run_training.py --config configs/owt/gpu-ts-setup-1e2.yaml --run-name gpu-ts-setup-1e2

echo "=== Starting run 2 of 2: gpu-initial-baseline-1e2 ==="
uv run python cs336_basics/run_training.py --config configs/owt/gpu-initial-baseline-1e2.yaml --run-name gpu-initial-baseline-1e2

echo "=== Sweep complete ==="

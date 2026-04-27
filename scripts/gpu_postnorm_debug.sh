#!/bin/bash

echo "=== Starting run 1 of 2: gpu-abln-postnorm-1e4.yaml ==="
uv run python cs336_basics/run_training.py --config configs/gpu-abln-postnorm-1e4.yaml --run-name gpu-postnorm-debug-1e4

echo "=== Starting run 2 of 2: gpu-abln-postnorm-1e2.yaml ==="
uv run python cs336_basics/run_training.py --config configs/gpu-abln-postnorm-1e2.yaml --run-name gpu-postnorm-debug-1e2

echo "=== Sweep complete ==="

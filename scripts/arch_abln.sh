#!/bin/bash

echo "=== Starting run 1 of 3: gpu-abln-nope-1e2.yaml ==="
uv run python cs336_basics/run_training.py --config configs/gpu-abln-nope-1e2.yaml

echo "=== Starting run 2 of 3: gpu-abln-silu-1e2.yaml ==="
uv run python cs336_basics/run_training.py --config configs/gpu-abln-silu-1e2.yaml

echo "=== Starting run 3 of 3: gpu-abln-postnorm-1e2.yaml ==="
uv run python cs336_basics/run_training.py --config configs/gpu-abln-postnorm-1e2.yaml

echo "=== Sweep complete ==="
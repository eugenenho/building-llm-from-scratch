#!/bin/bash

echo "=== Starting run 1 of 3: baseline-1e2.yaml ==="
uv run python cs336_basics/run_training.py --config configs/lr-sweep-1e2.yaml --run-name post-norm-debug-baseline-1e2

echo "=== Starting run 2 of 3: abln-postnorm-1e2.yaml ==="
uv run python cs336_basics/run_training.py --config configs/abln-postnorm-1e2.yaml --run-name post-norm-debug-abln-postnorm-1e2

echo "=== Starting run 3 of 3: abln-postnorm-1e4.yaml ==="
uv run python cs336_basics/run_training.py --config configs/abln-postnorm-1e4.yaml --run-name post-norm-debug-abln-postnorm-1e4

echo "=== Sweep complete ==="
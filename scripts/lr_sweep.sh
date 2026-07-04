#!/bin/bash
set -e

echo "=== Starting run 1 of 2: gpu-lr-sweep-6e2.yaml ==="
uv run python cs336_basics/main.py --config configs/gpu-lr-sweep-6e2.yaml

echo "=== Starting run 2 of 2: gpu-lr-sweep-3e3.yaml ==="
uv run python cs336_basics/main.py --config configs/gpu-lr-sweep-3e3.yaml

echo "=== Sweep complete ==="
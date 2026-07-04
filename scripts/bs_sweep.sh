#!/bin/bash

echo "=== Starting run 2 of 4: gpu-bs-sweep-64.yaml ==="
uv run python cs336_basics/main.py --config configs/gpu-bs-sweep-64.yaml

echo "=== Starting run 3 of 4: gpu-bs-sweep-1024.yaml ==="
uv run python cs336_basics/main.py --config configs/gpu-bs-sweep-1024.yaml

echo "=== Starting run 4 of 4: gpu-bs-sweep-16.yaml ==="
uv run python cs336_basics/main.py --config configs/gpu-bs-sweep-16.yaml

echo "=== Sweep complete ==="
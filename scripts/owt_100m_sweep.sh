#!/bin/bash

echo "=== Starting run 1 of 3: gpu-100m-lr-sweep-3e3 ==="
uv run python cs336_basics/main.py --config configs/owt/gpu-100m-lr-sweep-3e3.yaml

echo "=== Starting run 2 of 3: gpu-100m-lr-sweep-1e2 ==="
uv run python cs336_basics/main.py --config configs/owt/gpu-100m-lr-sweep-1e2.yaml

echo "=== Starting run 3of 3: gpu-100m-lr-sweep-3e2 ==="
uv run python cs336_basics/main.py --config configs/owt/gpu-100m-lr-sweep-3e2.yaml

echo "=== Sweep complete ==="


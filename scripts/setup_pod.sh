#!/bin/bash
# setup pod script for RunPod
set -e

cd /workspace/building-llm-from-scratch

# Install uv if missing
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# add to setup_pod.sh, after the uv install block:
 if ! command -v tmux &> /dev/null; then
    apt-get update && apt-get install -y tmux
fi

# Persist UV_PROJECT_ENVIRONMENT for future shells
grep -qxF 'export UV_PROJECT_ENVIRONMENT=/root/.venv' ~/.bashrc || \
echo 'export UV_PROJECT_ENVIRONMENT=/root/.venv' >> ~/.bashrc

export UV_PROJECT_ENVIRONMENT=/root/.venv
uv sync

# Set up SSH key for git (if backup exists on volume)
if [ -f /workspace/.ssh-backup/id_ed25519 ]; then
    mkdir -p /root/.ssh
    cp /workspace/.ssh-backup/id_ed25519* /root/.ssh/
    chmod 600 /root/.ssh/id_ed25519
    ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
else
    echo "Warning: no SSH key at /workspace/.ssh-backup/ — git auth will require credentials"
fi
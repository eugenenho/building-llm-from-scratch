# setup pod script for RunPod

#!/bin/bash
set -e

cd /workspace/building-llm-from-scratch

# Persist UV_PROJECT_ENVIRONMENT for future shells
grep -qxF 'export UV_PROJECT_ENVIRONMENT=/root/.venv' ~/.bashrc || \
echo 'export UV_PROJECT_ENVIRONMENT=/root/.venv' >> ~/.bashrc

export UV_PROJECT_ENVIRONMENT=/root/.venv
uv sync

mkdir -p /root/.ssh
cp /workspace/.ssh-backup/id_ed25519* /root/.ssh/
chmod 600 /root/.ssh/id_ed25519
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
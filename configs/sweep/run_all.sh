#!/usr/bin/env bash
# Roda as 10 configurações do sweep sequencialmente.
# Saídas em data/results/hybrid_sweep/<nome-da-config>/.
# Depois compare com: python configs/sweep/summarize.py
set -u
cd "$(dirname "$0")/../.."

for cfg in configs/sweep/[0-9][0-9]_*.yaml; do
    name=$(basename "$cfg" .yaml)
    echo "=== [$(date +%H:%M:%S)] Running $name ==="
    if ! python -m hybrid_stereo_method.hybrid.main --param_file "$cfg"; then
        echo "!!! $name FAILED — continuing with the next config" >&2
    fi
done
echo "=== [$(date +%H:%M:%S)] Sweep finished ==="

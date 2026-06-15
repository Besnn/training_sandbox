#!/usr/bin/env bash
# Launch the STN-FOMO feature-map inspector.
# Uses the existing pyenv sandbox env (has onnx/onnxruntime/opencv/flask).
set -e
cd "$(dirname "$0")"
source /Users/besnn/.pyenv/versions/3.11.14/envs/raspberry-pi-5-sandbox/bin/activate
exec python app.py

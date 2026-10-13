#!/bin/bash
LATEST=$(ls -td outputs/*/ | head -1)
python -m src.offline.colmap_runner \
  --keyframes "${LATEST}keyframes" \
  --out "${LATEST}colmap"

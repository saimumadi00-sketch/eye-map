#!/bin/bash
LATEST=$(ls -td outputs/*/ | head -1)
python src/offline/colmap_runner.py \
  --keyframes "${LATEST}keyframes" \
  --out "${LATEST}colmap"

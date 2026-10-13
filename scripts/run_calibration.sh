#!/bin/bash
python -m src.core.calibration --source 0 --out configs/calib.npz "$@"

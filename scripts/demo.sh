#!/bin/bash
# Quick 2D demo — runs in ~5 minutes on GPU
export LD_LIBRARY_PATH=/root/miniconda3/envs/asbs/lib:$LD_LIBRARY_PATH
cd /home/RESEARCH/REMAKE_ASBS
conda run -n asbs python -u train.py --experiment demo_2d "$@"

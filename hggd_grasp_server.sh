#!/usr/bin/env bash
# Launch the HGGD grasp-detection worker (torch+CUDA + RealSense D435).
# Run this in the conda env that has torch+CUDA+pytorch3d+pyrealsense2.
#   conda activate hggd-cu128
#   ./hggd_grasp_server.sh
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python hggd_grasp_server.py \
  --checkpoint-path ./HGGD_realsense_checkpoint \
  --host 127.0.0.1 \
  --port 6000 \
  --frame-id camera_color_optical_frame \
  --center-num 48 \
  --anchor-num 7 \
  --anchor-k 6 \
  --anchor-w 50 \
  --anchor-z 20 \
  --grid-size 8 \
  --all-points-num 25600 \
  --group-num 512 \
  --local-k 10 \
  --ratio 8 \
  --input-h 360 \
  --input-w 640 \
  --local-thres 0.01 \
  --heatmap-thres 0.01 \
  --rs-width 1280 \
  --rs-height 720 \
  --rs-fps 30

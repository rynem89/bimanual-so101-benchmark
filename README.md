# Bimanual SO-101 Benchmark

Simulation-first bimanual manipulation benchmark using NVIDIA Isaac Sim,
Isaac Lab, and LeIsaac.

## Task

Left SO-101 grasps a cube -> moves it to a handover region ->
right SO-101 grasps the cube -> left releases ->
right SO-101 places the cube into a target bin.

## Current milestone

Build the deterministic simulation foundation before any imitation learning.

Initial development target:

- two SO-101 follower arms
- simple table/workspace
- one cube
- articulation inspection
- independent left/right control

## Validated simulation stack

- Ubuntu 24.04.5 LTS
- NVIDIA driver 580.178.04 open
- CUDA 12.8.1
- Python 3.11.16
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.0
- LeIsaac 0.4.0

The simulator environment is treated as locked unless a demonstrated
compatibility problem requires a change.
#!/bin/bash

# ==========================================
# Session: gpu0
# ==========================================
tmux new-session -d -s gpu0 -n window1
tmux send-keys -t gpu0:window1 'conda activate datagenerator' C-m
tmux send-keys -t gpu0:window1 'python gen_data_new.py --gpu_id=0 --partition=000-000' C-m

tmux new-window -t gpu0 -n window2
tmux send-keys -t gpu0:window2 'conda activate datagenerator' C-m
tmux send-keys -t gpu0:window2 'python gen_data_new.py --gpu_id=0 --partition=000-001' C-m

# ==========================================
# Session: gpu1
# ==========================================
tmux new-session -d -s gpu1 -n window1
tmux send-keys -t gpu1:window1 'conda activate datagenerator' C-m
tmux send-keys -t gpu1:window1 'python gen_data_new.py --gpu_id=1 --partition=000-002' C-m

tmux new-window -t gpu1 -n window2
tmux send-keys -t gpu1:window2 'conda activate datagenerator' C-m
tmux send-keys -t gpu1:window2 'python gen_data_new.py --gpu_id=1 --partition=000-003' C-m

# ==========================================
# Session: gpu2
# ==========================================
# tmux new-session -d -s gpu2 -n window1
# tmux send-keys -t gpu2:window1 'conda activate datagenerator' C-m
# tmux send-keys -t gpu2:window1 'python gen_data_new.py --gpu_id=2 --partition=000-004' C-m

# tmux new-window -t gpu2 -n window2
# tmux send-keys -t gpu2:window2 'conda activate datagenerator' C-m
# tmux send-keys -t gpu2:window2 'python gen_data_new.py --gpu_id=2 --partition=000-005' C-m

# ==========================================
# Session: gpu3
# ==========================================
# tmux new-session -d -s gpu3 -n window1
# tmux send-keys -t gpu3:window1 'conda activate datagenerator' C-m
# tmux send-keys -t gpu3:window1 'python gen_data_new.py --gpu_id=3 --partition=000-006' C-m

# tmux new-window -t gpu3 -n window2
# tmux send-keys -t gpu3:window2 'conda activate datagenerator' C-m
# tmux send-keys -t gpu3:window2 'python gen_data_new.py --gpu_id=3 --partition=000-007' C-m

echo "All 4 GPU sessions and their windows have been started!"
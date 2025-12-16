#!/usr/bin/env bash

###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/root/zhaonankun/nuplan/dataset/nuplan-v1.1/splits/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/root/zhaonankun/nuplan/dataset/maps"                         # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/root/zhaonankun/Diffusion-Planner/train_npz"                  # preprocess training data

SELECTED_SCENARIOS_JSON="./selected_scenarios.json"                            # 已经划分好的 token 列表

NUM_WORKERS=20        # 外层并行进程数
TOKENS_PER_BATCH=10000  # 每个 shard 内一次 get_scenarios 处理多少个 token
###################################

python data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path  "$NUPLAN_MAP_PATH" \
  --save_path "$TRAIN_SET_PATH" \
  --scenario_token_file "$SELECTED_SCENARIOS_JSON" \
  --num_workers "$NUM_WORKERS" \
  --tokens_per_batch "$TOKENS_PER_BATCH"

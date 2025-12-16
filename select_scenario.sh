###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/root/zhaonankun/nuplan/dataset/nuplan-v1.1/splits/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/root/zhaonankun/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")
###################################

python select_scenario.py \
--data_path $NUPLAN_DATA_PATH \
--map_path  $NUPLAN_MAP_PATH \
--total_scenarios 1000000 \
--output_tokens ./selected_scenarios.json


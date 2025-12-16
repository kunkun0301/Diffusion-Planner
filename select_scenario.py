import os
import argparse
import json

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

from data_process import get_filter_parameters  # 直接复用你原来的函数


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select scenarios only')
    parser.add_argument('--data_path', required=True, type=str, help='path to raw data')
    parser.add_argument('--map_path', required=True, type=str, help='path to map data')

    parser.add_argument('--total_scenarios', type=int, default=1000000, help='number of scenarios to select')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    parser.add_argument(
        '--output_tokens',
        type=str,
        default='./selected_scenarios.json',
        help='where to save selected scenario tokens'
    )
    args = parser.parse_args()

    print("=== Stage 1: Loading log_names from nuplan_train.json ===")
    with open('./nuplan_train.json', "r", encoding="utf-8") as f:
        log_names = json.load(f)
    print(f"Loaded {len(log_names)} log_names")

    sensor_root = None
    db_files = None
    map_version = "nuplan-maps-v1.0"

    print("=== Stage 2: Building NuPlanScenarioBuilder ===")
    builder = NuPlanScenarioBuilder(
        args.data_path,
        args.map_path,
        sensor_root,
        db_files,
        map_version,
    )

    print("=== Stage 3: Creating ScenarioFilter ===")
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            num_scenarios_per_type=None,
            limit_total_scenarios=args.total_scenarios,
            shuffle=args.shuffle_scenarios,
            scenario_tokens=None,
            log_names=log_names,
        )
    )

    print("=== Stage 4: Querying scenarios from DB (this可能会比较久) ===")
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"[Done] Selected {len(scenarios)} scenarios")

    print("=== Stage 5: Collecting scenario tokens ===")
    scenario_tokens = []
    for s in scenarios:
        # 如果这里报错，就 print(dir(s)) 看一下属性名是 token 还是 scenario_token
        scenario_tokens.append(s.token)
    print(f"[Done] Collected {len(scenario_tokens)} tokens")

    print("=== Stage 6: Saving tokens to JSON ===")
    out_path = os.path.abspath(args.output_tokens)
    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(scenario_tokens, f, indent=2)

    print(f"[Done] Saved {len(scenario_tokens)} tokens to {out_path}")
    print("=== All stages finished. ===")

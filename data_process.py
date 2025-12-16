import os
import argparse
import json
import math
from multiprocessing import Process

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping


def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):
    """
    保持与原始版本一致，现在主要用 scenario_tokens 来指定要构建的场景。
    """
    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num)
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False         # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def process_shard(shard_idx: int, num_shards: int, all_tokens, args):
    """
    每个子进程处理一个 shard：
      1. 把本 shard 的 tokens 再拆成若干 batch
      2. 每个 batch 调用 builder.get_scenarios，并打印进度
      3. 调用 DataProcessor 处理并保存 npz（内部原有 tqdm 仍然保留）
    """
    total_tokens = len(all_tokens)
    shard_size = math.ceil(total_tokens / num_shards)
    start = shard_idx * shard_size
    end = min(total_tokens, (shard_idx + 1) * shard_size)
    scenario_tokens = all_tokens[start:end]

    print(
        f"[Shard {shard_idx}/{num_shards - 1}] "
        f"assigned tokens [{start}:{end}] -> {len(scenario_tokens)} tokens"
    )

    if len(scenario_tokens) == 0:
        print(f"[Shard {shard_idx}] no tokens to process, exiting.")
        return

    sensor_root = None
    db_files = None
    map_version = "nuplan-maps-v1.0"

    print(f"[Shard {shard_idx}] Building NuPlanScenarioBuilder...")
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)

    # 不再嵌套并行：内部串行，外层多进程
    worker = SingleMachineParallelExecutor(use_process_pool=False)

    # -------- 分 batch 查询，打印进度 --------
    tokens_per_batch = max(1, args.tokens_per_batch)
    num_batches = math.ceil(len(scenario_tokens) / tokens_per_batch)
    print(
        f"[Shard {shard_idx}] Querying scenarios in {num_batches} batches "
        f"(~{tokens_per_batch} tokens per batch)"
    )

    scenarios = []
    for batch_idx in range(num_batches):
        b_start = batch_idx * tokens_per_batch
        b_end = min(len(scenario_tokens), (batch_idx + 1) * tokens_per_batch)
        batch_tokens = scenario_tokens[b_start:b_end]

        print(
            f"[Shard {shard_idx}] Batch {batch_idx + 1}/{num_batches}: "
            f"querying {len(batch_tokens)} tokens "
            f"(global token idx {start + b_start}~{start + b_end - 1})"
        )

        scenario_filter = ScenarioFilter(
            *get_filter_parameters(
                num_scenarios_per_type=None,
                limit_total_scenarios=None,
                shuffle=args.shuffle_scenarios,
                scenario_tokens=batch_tokens,
                log_names=None,
            )
        )

        batch_scenarios = builder.get_scenarios(scenario_filter, worker)
        scenarios.extend(batch_scenarios)
        print(
            f"[Shard {shard_idx}] Batch {batch_idx + 1}/{num_batches} done: "
            f"got {len(batch_scenarios)} scenarios, total so far {len(scenarios)}"
        )

    print(f"[Shard {shard_idx}] Finished querying scenarios. Total scenarios: {len(scenarios)}")

    if len(scenarios) == 0:
        print(f"[Shard {shard_idx}] No scenarios constructed, exiting.")
        return

    # -------- 调用 DataProcessor（内部 tqdm 还在） --------
    print(f"[Shard {shard_idx}] Start processing data with DataProcessor on {len(scenarios)} scenarios...")
    del worker, builder, scenario_filter
    processor = DataProcessor(args)
    processor.work(scenarios)  # 这里原来的 tqdm 进度条会照常打印
    print(f"[Shard {shard_idx}] Data processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing (selected_scenarios.json + internal parallel)')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')

    # DataProcessor 相关参数，保持不变
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='(unused for selection now)')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios order')

    # 只使用 selected_scenarios.json
    parser.add_argument(
        '--scenario_token_file',
        type=str,
        default='./selected_scenarios.json',
        help='JSON file generated by select_scenarios.py, containing pre-selected scenario tokens (list)',
    )

    # 外层并行的 worker 数
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='number of parallel worker processes inside this script',
    )

    # 每个 shard 内部批量查询的 token 数
    parser.add_argument(
        '--tokens_per_batch',
        type=int,
        default=10000,
        help='number of tokens per get_scenarios batch inside each shard',
    )

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    # ============================================================
    # 1) 读取 selected_scenarios.json 里的 token 列表
    # ============================================================
    print("=== Stage 1: Loading pre-selected scenario tokens ===")
    with open(args.scenario_token_file, "r", encoding="utf-8") as f:
        all_tokens = json.load(f)

    if not isinstance(all_tokens, list):
        raise ValueError(
            f"{args.scenario_token_file} is expected to be a list of tokens, "
            f"but got type: {type(all_tokens)}"
        )

    total_tokens = len(all_tokens)
    print(f"Loaded {total_tokens} tokens from {args.scenario_token_file}")

    if total_tokens == 0:
        print("No tokens found. Nothing to do.")
        exit(0)

    num_workers = max(1, args.num_workers)
    num_workers = min(num_workers, total_tokens)  # 没必要比 token 数还多
    print(f"Using {num_workers} worker processes")

    # ============================================================
    # 2) 启动多个子进程，每个子进程处理一个 shard
    # ============================================================
    if num_workers == 1:
        # 单进程模式，方便调试
        process_shard(0, 1, all_tokens, args)
    else:
        procs = []
        for shard_idx in range(num_workers):
            p = Process(target=process_shard, args=(shard_idx, num_workers, all_tokens, args))
            p.start()
            procs.append(p)

        # 等所有子进程跑完
        for p in procs:
            p.join()

    # ============================================================
    # 3) 所有子进程跑完后，扫描 save_path 下的 npz 文件，生成训练列表
    # ============================================================
    print("=== Stage 3: Scanning generated .npz files ===")
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    npz_files = sorted(npz_files)

    output_list_path = './diffusion_planner_training.json'
    with open(output_list_path, 'w', encoding='utf-8') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names to {output_list_path}")
    print("=== All stages finished. ===")

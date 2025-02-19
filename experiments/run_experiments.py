import os
import subprocess
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with GPU scheduling.")
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Root directory containing experiment configurations.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        required=True,
        help="List of GPU IDs available for experiments.",
    )
    parser.add_argument(
        "--grace-period-factor",
        type=float,
        default=0.25,
        help="Grace period factor as a percentage of the maximum allowed time.",
    )
    return parser.parse_args()


def find_configs(base_dir):
    return glob(os.path.join(base_dir, "**", "config.yaml"), recursive=True)


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            return json.load(file)
    return {"completed": [], "timeout": [], "error": []}


def save_progress(progress, progress_file):
    with open(progress_file, "w") as file:
        json.dump(progress, file, indent=4)
    print(f"Progress saved to {progress_file}")


def calculate_timeout(config_path, grace_period_factor):
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # max_time = config.get("training.max_time", 3600)  # Default to 1 hour
    max_time = config["training"]["max_time"]
    timeout = max_time * (1 + grace_period_factor)
    return int(timeout)


def run_experiment(config_path, gpu_id, timeout_seconds, progress, progress_file):
    env = os.environ.copy()

    # Extract the directory and file name for Hydra
    config_dir = os.path.dirname(config_path)
    config_file = os.path.basename(config_path).replace(".yaml", "")

    cmd = [
        "python",
        "run_experiment_hydra.py",
        f"--config-path={config_dir}",
        f"--config-name={config_file}",
        f"hydra.run.dir={config_dir}",
        f"+device={gpu_id}",
    ]

    print(f"Running: {' '.join(cmd)} on GPU {gpu_id} with timeout {timeout_seconds}s")

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode == 0:
            progress["completed"].append(config_path)
            print(f"Experiment completed: {config_path}")
        else:
            progress["error"].append(config_path)
            print(f"Experiment failed: {config_path}")
            print(f"STDOUT:\n{stdout.decode()}")
            print(f"STDERR:\n{stderr.decode()}")

    except subprocess.TimeoutExpired:
        print(
            f"Experiment at {config_path} exceeded the timeout of {timeout_seconds}s. \
                Terminating."
        )
        process.terminate()
        progress["timeout"].append(config_path)

    except Exception as e:
        print(f"Error occurred while running experiment {config_path}: {e}")
        progress["error"].append(config_path)

    finally:
        save_progress(progress, progress_file)


def run_all_experiments(base_dir, devices, grace_period_factor, progress_file):
    configs = find_configs(base_dir)
    progress = load_progress(progress_file)

    gpu_queue = Queue()
    for gpu_id in devices:
        gpu_queue.put(gpu_id)

    def run_config(config_path):
        if (
            config_path in progress["completed"]
            or config_path in progress["timeout"]
            or config_path in progress["error"]
        ):
            print(f"Skipping already processed experiment: {config_path}")
            return

        gpu_id = gpu_queue.get()
        try:
            timeout_seconds = calculate_timeout(config_path, grace_period_factor)
            run_experiment(
                config_path, gpu_id, timeout_seconds, progress, progress_file
            )
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        executor.map(run_config, configs)


if __name__ == "__main__":
    args = parse_args()
    progress_file = os.path.join(args.base_dir, "progress.json")
    run_all_experiments(
        args.base_dir, args.devices, args.grace_period_factor, progress_file
    )

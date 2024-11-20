import os
import subprocess
import json
from glob import glob

# Configuration
BASE_DIR = "performance_full_krr"  # Root directory containing experiment configurations
DEVICES = [0, 1, 2, 3]  # List of GPU IDs available for experiments
GRACE_PERIOD_FACTOR = 0.25  # 25% additional time as a grace period
PROGRESS_FILE = "progress.json"  # File to track completed experiments


def find_configs(base_dir):
    """
    Find all config.yaml files in the folder structure.
    :param base_dir: Base directory to search for configurations.
    :return: List of paths to config.yaml files.
    """
    return glob(os.path.join(base_dir, "**", "config.yaml"), recursive=True)


def load_progress(progress_file):
    """
    Load progress from a JSON file.
    :param progress_file: Path to the progress tracking file.
    :return: Dictionary of completed experiments.
    """
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            return json.load(file)
    return {}


def save_progress(progress, progress_file):
    """
    Save progress to a JSON file.
    :param progress: Dictionary of completed experiments.
    :param progress_file: Path to the progress tracking file.
    """
    with open(progress_file, "w") as file:
        json.dump(progress, file, indent=4)


def calculate_timeout(config_path, grace_period_factor):
    """
    Calculate the timeout for an experiment based on its config.yaml.
    :param config_path: Path to the config.yaml file.
    :param grace_period_factor: Grace period as a fraction of max training time.
    :return: Calculated timeout in seconds.
    """
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    max_time = config.get(
        "training.max_time", 3600
    )  # Default to 1 hour if not specified
    timeout = max_time * (1 + grace_period_factor)
    return int(timeout)


def run_experiment(config_path, gpu_id, timeout_seconds, progress):
    """
    Run a single experiment with a timeout.
    :param config_path: Path to the config.yaml file.
    :param gpu_id: GPU ID to use.
    :param timeout_seconds: Calculated timeout in seconds.
    :param progress: Dictionary to track experiment progress.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Set GPU for this subprocess

    # Override device in the experiment
    cmd = [
        "python",
        "run_experiment_hydra.py",
        f"hydra.run.dir={os.path.dirname(config_path)}",
        f"+config={config_path}",
        f"device={gpu_id}",  # Dynamically set device as a command-line override
    ]

    print(f"Running: {' '.join(cmd)} on GPU {gpu_id} with timeout {timeout_seconds}s")

    try:
        process = subprocess.Popen(cmd, env=env)
        process.wait(timeout=timeout_seconds)
        progress[config_path] = "completed"  # Mark as completed if successful
        save_progress(progress, PROGRESS_FILE)
        print(f"Experiment completed: {config_path}")

    except subprocess.TimeoutExpired:
        print(
            f"Experiment at {config_path} exceeded the timeout of {timeout_seconds}s. \
                  Terminating."
        )
        process.terminate()
        process.wait()
        progress[config_path] = "timeout"  # Mark as timeout
        save_progress(progress, PROGRESS_FILE)

    except Exception as e:
        print(f"Error occurred while running experiment {config_path}: {e}")
        progress[config_path] = "error"  # Mark as error
        save_progress(progress, PROGRESS_FILE)


def run_all_experiments(base_dir, devices, grace_period_factor, progress_file):
    """
    Run all experiments for configurations in the base directory.
    :param base_dir: Base directory containing experiment configurations.
    :param devices: List of GPU IDs to use.
    :param grace_period_factor: Grace period as a fraction of max training time.
    :param progress_file: File to track completed experiments.
    """
    configs = find_configs(base_dir)
    progress = load_progress(progress_file)
    processes = []

    for i, config_path in enumerate(configs):
        # Skip completed experiments
        if progress.get(config_path, "") in ["completed", "timeout", "error"]:
            print(f"Skipping {progress[config_path]} experiment: {config_path}")
            continue

        # Calculate timeout based on the grace period
        timeout_seconds = calculate_timeout(config_path, grace_period_factor)

        # Assign GPU in a round-robin fashion
        gpu_id = devices[len(processes) % len(devices)]

        # Run the experiment
        run_experiment(config_path, gpu_id, timeout_seconds, progress)

        # Manage GPU limits
        if len(processes) >= len(devices):
            for process, path in processes:
                process.wait()
            processes = []

    # Ensure all remaining processes complete
    for process, path in processes:
        process.wait()


if __name__ == "__main__":
    run_all_experiments(BASE_DIR, DEVICES, GRACE_PERIOD_FACTOR, PROGRESS_FILE)

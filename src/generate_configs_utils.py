import os
import yaml


def add_kernel_params(config, kernel_configs):
    kernel_params = kernel_configs[config["dataset"]]
    config["kernel"] = {
        "type": kernel_params["type"],
        "sigma": kernel_params["sigma"],
    }
    if "nu" in kernel_params:
        config["kernel"]["nu"] = kernel_params["nu"]


def save_configs(combinations, output_dir):
    for idx, combo in enumerate(combinations):
        folder_path = os.path.join(
            output_dir, combo["dataset"], combo["model"], f"config_{idx}"
        )
        os.makedirs(folder_path, exist_ok=True)

        config_path = os.path.join(folder_path, "config.yaml")
        with open(config_path, "w") as file:
            yaml.dump(combo, file, default_flow_style=False)

        print(f"Generated: {config_path}")

import os
import yaml

from collections import OrderedDict


def update_yaml_file(file_path, key_path, value):
    with open(file_path, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    # Dynamically set the value based on the provided key path
    set_nested_value(config, key_path, value)

    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


def set_nested_value(dictionary, key_path, value):
    keys = key_path.split(".")
    d = dictionary
    for key in keys[:-1]:
        d = d.setdefault(key, OrderedDict())
    d[keys[-1]] = value


def update_yaml_files(cfg) -> None:
    static_rgb_height = cfg.static_rgb_shape[0]
    static_rgb_width = cfg.static_rgb_shape[1]
    gripper_rgb_height = cfg.gripper_rgb_shape[0]
    gripper_rgb_width = cfg.gripper_rgb_shape[1]
    tactile_sensor_height = cfg.tactile_sensor_shape[0]
    tactile_sensor_width = cfg.tactile_sensor_shape[1]

    # Update the camera configurations
    dataset_configs = os.path.join(cfg.dataset_path, "validation", ".hydra")
    config_yaml_path = os.path.join(dataset_configs, "config.yaml")
    merged_config_path = os.path.join(dataset_configs, "merged_config.yaml")

    # Update the camera configurations for the simulation environment
    camera_configs = os.path.join(
        os.pardir, "calvin-sim", "calvin_env", "conf", "cameras", "cameras"
    )
    static_path = os.path.join(camera_configs, "static.yaml")
    gripper_path = os.path.join(camera_configs, "gripper.yaml")
    tactile_path = os.path.join(camera_configs, "tactile.yaml")
    opposing_path = os.path.join(camera_configs, "opposing.yaml")

    update_dict = {
        config_yaml_path: {
            "cameras.static.width": static_rgb_width,
            "cameras.static.height": static_rgb_height,
            "cameras.gripper.width": gripper_rgb_width,
            "cameras.gripper.height": gripper_rgb_height,
            "cameras.tactile.width": tactile_sensor_width,
            "cameras.tactile.height": tactile_sensor_height,
        },
        merged_config_path: {
            "cameras.static.width": static_rgb_width,
            "cameras.static.height": static_rgb_height,
            "cameras.gripper.width": gripper_rgb_width,
            "cameras.gripper.height": gripper_rgb_height,
            "cameras.tactile.width": tactile_sensor_width,
            "cameras.tactile.height": tactile_sensor_height,
        },
        static_path: {
            "width": static_rgb_width,
            "height": static_rgb_height,
        },
        gripper_path: {
            "width": gripper_rgb_width,
            "height": gripper_rgb_height,
        },
        tactile_path: {
            "width": tactile_sensor_width,
            "height": tactile_sensor_height,
        },
        opposing_path: {
            "width": static_rgb_width,
            "height": static_rgb_height,
        },
    }

    for path, yaml_dict in update_dict.items():
        for key, value in yaml_dict.items():
            update_yaml_file(path, key, value)

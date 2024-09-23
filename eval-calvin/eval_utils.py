import os
import cv2
import numpy as np
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


def gather_renders(cfg, env):
    images = []

    assert cfg.vis_rgb_static, "At least one RGB image must be visualized"
    rgb_static = env.render_image("rgb_static")
    images.append(rgb_static)

    if cfg.vis_rgb_gripper:
        rgb_gripper = env.render_image("rgb_gripper")
        rgb_gripper = cv2.resize(
            rgb_gripper, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        images.append(rgb_gripper)

    if cfg.vis_rgb_tactile:
        rgb_tactile = env.render_image("rgb_tactile")
        # Split tactile RGB into 2 images (3 channels each)
        rgb_tactile_1 = rgb_tactile[:, :, :3]
        rgb_tactile_2 = rgb_tactile[:, :, 3:]
        rgb_tactile_1 = cv2.resize(
            rgb_tactile_1, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        rgb_tactile_2 = cv2.resize(
            rgb_tactile_2, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        images.append(rgb_tactile_1)
        images.append(rgb_tactile_2)

    if cfg.vis_depth_static:
        depth_static = env.render_image("depth_static")
        depth_resized = cv2.resize(
            depth_static, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        images.append(depth_colormap)

    if cfg.vis_depth_gripper:
        depth_gripper = env.render_image("depth_gripper")
        depth_resized = cv2.resize(
            depth_gripper, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        images.append(depth_colormap)

    if cfg.vis_depth_tactile:
        depth_tactile = env.render_image("depth_tactile")
        # Split tactile depth into 2 depth images
        depth_tactile_1 = depth_tactile[:, :, 0]
        depth_tactile_2 = depth_tactile[:, :, 1]
        depth_resized_1 = cv2.resize(
            depth_tactile_1, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        depth_resized_2 = cv2.resize(
            depth_tactile_2, rgb_static.shape[:2], interpolation=cv2.INTER_LINEAR
        )
        depth_normalized_1 = cv2.normalize(
            depth_resized_1, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_normalized_1 = depth_normalized_1.astype(np.uint8)
        depth_colormap_1 = cv2.applyColorMap(depth_normalized_1, cv2.COLORMAP_JET)
        depth_normalized_2 = cv2.normalize(
            depth_resized_2, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_normalized_2 = depth_normalized_2.astype(np.uint8)
        depth_colormap_2 = cv2.applyColorMap(depth_normalized_2, cv2.COLORMAP_JET)
        images.append(depth_colormap_1)
        images.append(depth_colormap_2)

    rows = [np.hstack(images[i : i + 4]) for i in range(0, len(images), 4)]
    final_image = np.vstack(rows)
    return final_image

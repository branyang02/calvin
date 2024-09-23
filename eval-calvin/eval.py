from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import imageio
from tqdm.auto import tqdm
import tyro

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from eval_utils import update_yaml_files

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from calvin_agent.evaluation.rich_utils import CONSOLE
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored

from calvin_env.envs.play_table_env import get_env
from calvin_env.envs.play_table_env import PlayTableSimEnv

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 10


@dataclass
class Config:
    dataset_path: str = (
        "../calvin-datasets/mini_dataset"  # Path to the dataset root directory
    )
    static_rgb_shape: tuple = (200, 200)  # height, width of the static RGB image
    gripper_rgb_shape: tuple = (84, 84)  # height, width of the gripper RGB image
    tactile_sensor_shape: tuple = (
        160,
        120,
    )  # height, width of the tactile sensor image
    debug: bool = False  # Print debug info and visualize environment.


def make_env(dataset_path):
    val_folder = os.path.join(dataset_path, "validation")
    env = get_env(val_folder, show_gui=False)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, obs, goal):
        # Random action
        action_displacement = np.random.uniform(low=-1, high=1, size=6)
        action_gripper = np.random.choice([-1, 1], size=1)
        action = np.concatenate((action_displacement, action_gripper), axis=-1)
        return action


class CalvinEvaluator:

    def __init__(self, model: CalvinBaseModel, env: PlayTableSimEnv, cfg: Config):
        self.model = model
        self.env = env
        self.cfg = cfg

        self.dataset_name = Path(cfg.dataset_path).name

    def evaluate_policy(self):
        # Load task oracle and validation annotations
        conf_dir = (
            Path(__file__).absolute().parents[1] / "calvin-sim/calvin_models/conf"
        )
        task_cfg = OmegaConf.load(
            conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
        )
        task_oracle = hydra.utils.instantiate(task_cfg)
        val_annotations = OmegaConf.load(
            conf_dir / "annotations/new_playtable_validation.yaml"
        )

        eval_log_dir = get_log_dir(f"{self.dataset_name}_eval_results")
        eval_sequences = get_sequences(NUM_SEQUENCES)

        results = []

        if not self.cfg.debug:
            eval_sequences = tqdm(eval_sequences, position=0, leave=True)

        for initial_state, eval_sequence in eval_sequences:
            result = self.evaluate_sequence(
                task_oracle,
                initial_state,
                eval_sequence,
                val_annotations,
            )
            results.append(result)
            if not self.cfg.debug:
                eval_sequences.set_description(
                    " ".join(
                        [
                            f"{i + 1}/5 : {v * 100:.1f}% |"
                            for i, v in enumerate(count_success(results))
                        ]
                    )
                    + "|"
                )
        print_and_save(results, eval_sequences, eval_log_dir)

        return results

    def evaluate_sequence(
        self, task_checker, initial_state, eval_sequence, val_annotations
    ):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        if self.cfg.debug:
            CONSOLE.print(
                f"[bright_cyan]Evaluating sequence: {' -> '.join(eval_sequence)}"
            )
        for subtask in eval_sequence:
            success = self.rollout(
                task_checker,
                subtask,
                val_annotations,
            )
            if success:
                success_counter += 1
            else:
                return success_counter
        return success_counter

    def rollout(self, task_oracle, subtask, val_annotations):
        if self.cfg.debug:
            CONSOLE.print(f"[bright_cyan]Subtask: {subtask} ")
            canvas_all = []
        obs = self.env.get_obs()

        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        CONSOLE.print(f"[cyan]Lang annotation: {lang_annotation}")
        self.model.reset()
        start_info = self.env.get_info()

        for step in range(EP_LEN):
            action = self.model.step(obs, lang_annotation)
            obs, _, _, current_info = self.env.step(action)
            if self.cfg.debug:
                img = self.env.render(mode="all")
                canvas_all.append(img)

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )
            if len(current_task_info) > 0:
                if self.cfg.debug:
                    self._save_video(canvas_all, subtask)
                    CONSOLE.print(f"[bold green]success")
                return True

        # save to video
        if self.cfg.debug:
            self._save_video(canvas_all, subtask)
            CONSOLE.print(f"[bold red]fail")
        return False

    def _save_video(self, canvas_all, subtask):
        video_dir = f"{self.dataset_name}_videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(
            f"{video_dir}/traj_{subtask}.mp4", fps=30, macro_block_size=1
        )
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        CONSOLE.print(f"[yellow]Video saved to {video_dir}/traj_{subtask}.mp4")


def main():
    seed_everything(0, workers=True)  # type:ignore
    cfg = tyro.cli(Config)
    update_yaml_files(cfg)  # updates rendering resolutions in yaml files

    model = CustomModel()
    env = make_env(cfg.dataset_path)
    evaluator = CalvinEvaluator(model, env, cfg)
    evaluator.evaluate_policy()


if __name__ == "__main__":
    main()

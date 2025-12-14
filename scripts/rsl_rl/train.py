# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)



#----------------------------- main training imports ---------------------------------



"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import amphrobot.tasks  # noqa: F401
from amphrobot.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

#----------------------------- main training block ---------------------------------

class DebugActionWrapper(gym.Wrapper):
    def __init__(self, env, every=100):
        super().__init__(env)
        self._dbg_i = 0
        self._every = every

    def step(self, action):
        # action here is what gets passed down into the IsaacLab env (often after rsl-rl clipping)
        if self._dbg_i % self._every == 0:
            a = action
            if isinstance(a, torch.Tensor):
                amin = a.min().item()
                amax = a.max().item()
                amean = a.mean().item()
                astd = a.std(unbiased=False).item()
                print(f"[DEBUG] step {self._dbg_i} | action range [{amin:.3f}, {amax:.3f}] mean {amean:.3f} std {astd:.3f}")
                print(f"[DEBUG] first env action[:12]: {a[0, :].detach().cpu()}")
            else:
                # fallback for numpy
                import numpy as np
                amin, amax = np.min(a), np.max(a)
                print(f"[DEBUG] step {self._dbg_i} | action range [{amin:.3f}, {amax:.3f}]")
                print(f"[DEBUG] first env action[:10]: {a[0, :10]}")
        self._dbg_i += 1
        return self.env.step(action)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # # --- DEBUG ROOT STATE ---
    # isaac_env = env.unwrapped  # unwrap OrderEnforcing / Gymnasium wrappers

    # print("[DEBUG] isaac_env type:", type(isaac_env))

    # # reset the raw IsaacLab env
    # obs, _ = isaac_env.reset()

    # # read root state before step
    # root_before = isaac_env.scene["robot"].data.root_state_w.clone()
    # print("[DEBUG] root before step (x, y, z):", root_before[0, :3])

    # # get action dimension from action manager
    # action_dim = isaac_env.action_manager.total_action_dim

    # # create a dummy zero action with the right shape and type
    # dummy_action = torch.zeros(
    #     (isaac_env.num_envs, action_dim),
    #     device=isaac_env.device,
    # )

    # for i in range(20):
    #     obs, rew, terminated, truncated, info = isaac_env.step(dummy_action)
    #     root = isaac_env.scene["robot"].data.root_state_w.clone()
    #     print(f"[DEBUG] step {i:2d} (x, y, z):", root[0, :3])

    # # stop here so training doesn't run during debugging
    # sys.exit(0)
    # # --- END DEBUG ---

    # isaac_env = env.unwrapped   # unwrap Gym wrapper

    # # Reset environment
    # obs, _ = isaac_env.reset()

    # # Access the contact sensor configured in your env
    # # (in your cfg it is named "contact_forces")
    # contact_sensor = isaac_env.scene.sensors["contact_forces"]

    # # Inspect available bodies
    # body_names = contact_sensor.body_names
    # print("[DEBUG] contact sensor bodies:")
    # for i, name in enumerate(body_names):
    #     print(f"  {i}: {name}")

    # # Find the index of the base link in the contact sensor bodies
    # base_index = None
    # for i, name in enumerate(body_names):
    #     if "base_link" in name:   # adapt if your base link has a slightly different name
    #         base_index = i
    #         break

    # if base_index is None:
    #     print("[ERROR] Could not find 'base_link' in contact sensor body_names")
    #     sys.exit(1)

    # print(f"[DEBUG] Using base_index = {base_index} for {body_names[base_index]}")

    # # Number of debug steps to run
    # num_debug_steps = 50

    # # Step through physics with zero action
    # action_dim = isaac_env.action_manager.total_action_dim
    # dummy_action = torch.zeros((isaac_env.num_envs, action_dim), device=isaac_env.device)

    # print("\n[DEBUG] Printing base-link contact forces...")

    # for i in range(num_debug_steps):
    #     isaac_env.step(dummy_action)

    #     # Shape: (num_envs, history_length, num_bodies, 3)
    #     forces_hist = contact_sensor.data.net_forces_w_history

    #     # Forces for base body: (num_envs, history_length, 3)
    #     base_forces = forces_hist[:, :, base_index, :]

    #     # Magnitudes: (num_envs, history_length)
    #     force_mag = torch.norm(base_forces, dim=-1)

    #     # Max over history
    #     max_force = force_mag.max(dim=1)[0]   # (num_envs,)

    #     print(f"Step {i:02d} | Max base contact force: {max_force[0].item():.2f} N")

    # sys.exit(0)


    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # env = DebugActionWrapper(env, every=100)

    # wrap around environment for rsl-rl
   
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

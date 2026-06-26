#!/usr/bin/env python

import numpy as np
import torch
import cv2
import sys
import random
import json
import matplotlib.pyplot as plt 
if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import os
from tqdm import tqdm
import atexit
import logging
import multiprocessing
import sys
import traceback
import torch.multiprocessing as mp 

import igibson
from igibson.envs.ma_env_nav import Ma_Nav
from igibson.envs.multi_parallel_env import ParallelNavEnv,VecFlattenObs

import setproctitle
from pathlib import Path
from igibson.onpolicy.config_mappo import get_config
from igibson.utils.utils import parse_config


import builtins
import inspect

_real_print = builtins.print

def traced_print(*args, **kwargs):
    frame = inspect.currentframe().f_back  # caller
    info = inspect.getframeinfo(frame)
    _real_print(f"[PRINT from {info.filename}:{info.lineno} in {info.function}]")
    _real_print(*args, **kwargs)
def make_env_fns_from_config(config_filename: str, device_idx: int = 0):
    """
    Reads config, collects scene IDs, returns a list of zero-arg callables.
    Each callable constructs one Ma_Nav for one scene_id.
    """
    config = parse_config(config_filename)

    scene_ids = config.get("scene_id", [])
    if isinstance(scene_ids, (str, int)):
        scene_ids = [scene_ids]
    elif scene_ids is None:
        scene_ids = []

    mode = config.get("mode")

    def _make_one(scene_id):
        def _thunk():
            return Ma_Nav(
                config_file=config_filename,
                scene_id=scene_id,
                mode=mode,
                device_idx=device_idx,
            )
        return _thunk

    return [_make_one(sid) for sid in scene_ids]




def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='specific_goal_two_bots', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = True
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")
    

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)



    logging.basicConfig(level=logging.INFO)
    config_filename = os.path.join(os.path.dirname(igibson.__file__), 'configs/turtlebot_nav_train.yaml')
    envs_load = make_env_fns_from_config(config_filename, device_idx=0)
    envs = ParallelNavEnv(envs_load[:all_args.n_rollout_threads], blocking=False)
    envs=  VecFlattenObs(envs, cuda=True)



    
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))


    
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    device = torch.device("cuda:0")
    num_agents = all_args.num_agents
    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from igibson.onpolicy.runner.shared.multi_robot import Multi_robot as Runner
    else:
        from igibson.onpolicy.runner.separated.multi_robot import Multi_robot as Runner
    
    runner = Runner(config)
    
    runner.run()
 

if __name__ == "__main__":
    main(sys.argv[1:])

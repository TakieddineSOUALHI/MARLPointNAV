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

import igibson

from igibson.envs.ma_env_nav import Ma_Nav
from igibson.envs.mutli_parallel_env import ParallelNavEnv,VecFlattenObs
import torchvision.transforms.functional as F


#import wandb
#import socket
import setproctitle
from pathlib import Path
from igibson.onpolicy.config import get_config
from igibson.onpolicy.envs.mpe.MPE_env import MPEEnv
from igibson.onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from igibson.utils.utils import parse_config


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='specific_goal_two_bots', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all_args",all_args.eval_episodes)
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
    
    '''
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        
        #torch.set_num_threads(all_args.n_training_threads)
        #if all_args.cuda_deterministic:
        #    torch.backends.cudnn.benchmark = False
        #    torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        #torch.set_num_threads(all_args.n_training_threads)'''



    logging.basicConfig(level=logging.INFO)
    print("IGIBSON path",os.path.join(os.path.dirname(igibson.__file__), 'configs/turtlebot_nav_test.yaml'))
    config_filename =os.path.join(os.path.dirname(igibson.__file__), 'configs/turtlebot_nav_test.yaml')#os.path.join(os.path.dirname(igibson.__file__), "..", "tests", "test.yaml")
    print("config file",config_filename)
    config=parse_config(config_filename)
    
    def load_env_0():
        return Ma_Nav(config_file=config_filename,scene_id=config['scene_id'][0], mode=config['mode'],device_idx=3)
    '''
    def load_env_1():
        return Ma_Nav(config_file=config_filename,scene_id=config['scene_id'][1], mode=config['mode'],device_idx=3)
    
    def load_env_2():
        return Ma_Nav(config_file=config_filename,scene_id=config['scene_id'][2], mode=config['mode'],device_idx=3)
    def load_env_3():
        return Ma_Nav(config_file=config_filename,scene_id=config['scene_id'][3], mode=config['mode'],device_idx=3)
    def load_env_4():
        return Ma_Nav(config_file=config_filename,scene_id=config['scene_id'][4], mode=config['mode'],device_idx=3)'''
   
   
   
    envs_load=[load_env_0]#,load_env_1]#,load_env_2,load_env_3,load_env_4]

    envs = ParallelNavEnv(envs_load, blocking=False)
    envs= VecFlattenObs(envs, cuda=True)
   
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
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
    #torch.manual_seed(all_args.seed)
    #torch.cuda.manual_seed_all(all_args.seed)
    #np.random.seed(all_args.seed)

    # env init
    device = torch.device("cuda:3")

    #envs = make_train_env(all_args)
    #eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    eval_envs=envs
    num_agents = all_args.num_agents
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from igibson.onpolicy.runner.shared.bot_runner import BOTRunner as Runner
        print("Runner imported")
    else:
        from igibson.onpolicy.runner.separated.bot_runner import BOTRunner as Runner
    
    
    runner = Runner(config)
    
    runner.render()
    '''
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()'''


if __name__ == "__main__":
    main(sys.argv[1:])

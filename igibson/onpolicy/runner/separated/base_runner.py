import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from igibson.onpolicy.utils.separated_buffer import SeparatedDictReplayBuffer
from igibson.onpolicy.utils.util import get_shape_from_obs_space
from gym import spaces
from collections import OrderedDict
import imageio


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.eval_episodes=self.all_args.eval_episodes
        self.use_state=self.all_args.use_state
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        from igibson.onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from igibson.onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

            
        self.policy = []
    
        if self.use_centralized_V: 

            share_observation_space =  [self.envs.share_observation_space for _ in range(self.num_agents)]
            observation_space=[self.envs.observation_space for _ in range(self.num_agents)]
            self.action_space= [self.envs.action_space for _ in range(self.num_agents)]

        elif not self.use_centralized_V: 

            share_observation_space = [self.envs.observation_space for _ in range(self.num_agents)] 
            observation_space=[self.envs.observation_space for _ in range(self.num_agents)]
            self.action_space= [self.envs.action_space for _ in range(self.num_agents)]

        for agent_id in range(self.num_agents):

            # policy network
            po = Policy(self.all_args,
                        observation_space[agent_id],
                        share_observation_space[agent_id],
                        self.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)
        

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
           
            # buffer      
            bu = SeparatedDictReplayBuffer(self.all_args,
                                       observation_space[agent_id],
                                       share_observation_space[agent_id],
                                       self.action_space[agent_id],device = self.device)

            self.buffer.append(bu)
            self.trainer.append(tr)

        if self.model_dir is not None:
            self.restore()    

        


    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            shared_obs=OrderedDict()

            for key in self.buffer[agent_id].share_obs.keys():
                shared_obs[key]=self.buffer[agent_id].share_obs[key][-1]

            next_value = self.trainer[agent_id].policy.get_values(shared_obs, 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
                
            self.buffer[agent_id].compute_returns(next_value.to(self.device), self.trainer[agent_id].value_normalizer)
           
    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            actor_optimizer=self.trainer[agent_id].policy.actor_optimizer
            torch.save(actor_optimizer.state_dict(), str(self.save_dir) + "/actor_optimizer" + str(agent_id) + ".pt")
            critic_optimizer=self.trainer[agent_id].policy.critic_optimizer
            torch.save(critic_optimizer.state_dict(), str(self.save_dir) + "/critic_optimizer" + str(agent_id) + ".pt")


            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")
            

    def restore(self):
        for agent_id in range(self.num_agents):
            
           
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            for param_tensor in policy_actor_state_dict:
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
          
                if self.all_args.use_valuenorm:
                    policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                    self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

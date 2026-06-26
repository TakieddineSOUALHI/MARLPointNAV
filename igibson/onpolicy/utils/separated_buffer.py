import torch
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from igibson.onpolicy.utils.util import check, get_shape_from_obs_space, get_shape_from_act_space
from gym import spaces
import matplotlib.pyplot as plt 


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


def get_obs_shape(observation_space):
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()} 


class SeparatedDictReplayBuffer(object):
    def __init__(self, args, obs_space, cent_obs_space, act_space,device):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.device=device
        obs_shape = get_obs_shape(obs_space)
        share_obs_shape= get_obs_shape(cent_obs_space)

        self.share_obs=OrderedDict()
        self.obs=OrderedDict()


        for key, obs_input_shape in share_obs_shape.items(): 
            self.share_obs[key] = torch.zeros((self.episode_length + 1, self.n_rollout_threads, *get_shape_from_obs_space(cent_obs_space[key])),device=self.device)

        for key, obs_input_shape in obs_shape.items(): 
            self.obs[key] = torch.zeros((self.episode_length + 1, self.n_rollout_threads, *get_shape_from_obs_space(obs_space[key])),device=self.device)




        self.rnn_states = torch.zeros((self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size),device=self.device)
        self.rnn_states_critic =  torch.zeros((self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size),device=self.device)

        self.value_preds = torch.zeros((self.episode_length + 1, self.n_rollout_threads, 1),device=self.device)
        self.returns = torch.zeros((self.episode_length + 1, self.n_rollout_threads, 1),device=self.device)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = torch.zeros((self.episode_length, self.n_rollout_threads, act_shape), device=self.device)
        self.action_log_probs = torch.zeros((self.episode_length, self.n_rollout_threads, act_shape), device=self.device)
        self.rewards = torch.zeros((self.episode_length, self.n_rollout_threads, 1),device=self.device)
        
        self.masks = torch.ones((self.episode_length + 1, self.n_rollout_threads, 1),device=self.device)
        self.bad_masks = torch.ones((self.episode_length + 1, self.n_rollout_threads, 1),device=self.device)
        self.active_masks = torch.ones((self.episode_length + 1, self.n_rollout_threads, 1),device=self.device)
        self.step = 0

        
    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        #print("Type and size",share_obs['task_obs'].shape,type(share_obs['task_obs']))
        for key in obs.keys():
            self.obs[key][self.step + 1].copy_(obs[key])

        for key in share_obs.keys(): 
            self.share_obs[key][self.step + 1].copy_(share_obs[key])
        
        self.rnn_states[self.step + 1].copy_(rnn_states)
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)
        self.actions[self.step].copy_(actions)#.copy()
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
     
        
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length


    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        

        self.share_obs[key][self.step] = share_obs[key].copy()
        self.share_obs[key][self.step] = share_obs[key].copy()

        self.obs[key][self.step] = obs[key].copy()
        self.obs[key][self.step] = obs[key].copy()

        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    



    def after_update(self):
        
        for key in self.share_obs.keys():
            self.share_obs[key][0].copy_(self.share_obs[key][-1])
        for key in self.obs.keys():
            self.obs[key][0].copy_(self.obs[key][-1])

        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.active_masks[0].copy_(self.active_masks[-1])
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()



    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()



    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]




    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch



    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch=defaultdict(list)
            obs_batch=defaultdict(list)
    
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            #value_preds_others_batch =[]
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for key in self.obs.keys():
                    obs_batch[key].append(self.obs[key][:-1, ind])
                    
                for key in self.share_obs.keys():
                    share_obs_batch[key].append(self.share_obs[key][:-1, ind])

                    #share_obs_batch_2.append(self.share_obs[:-1, ind])
                    #obs_batch_2.append(self.obs[:-1, ind])
                
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                #value_preds_others_batch.append(self.value_preds_others[:-1,ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            for key in self.obs.keys():
                obs_batch[key]=torch.stack(obs_batch[key],1)

            for key in self.share_obs.keys():
                share_obs_batch[key]=torch.stack(share_obs_batch[key],1)

        
            actions_batch = torch.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            #value_preds_others_batch= torch.stack(value_preds_others_batch)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            active_masks_batch = torch.stack(active_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = torch.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            
            for key in self.obs.keys():
                obs_batch[key]= _flatten(T, N,obs_batch[key])
            
            for key in self.share_obs.keys():
                share_obs_batch[key]= _flatten(T, N, share_obs_batch[key])

                
            #share_obs_batch = _flatten(T, N, share_obs_batch)
            #obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            #value_preds_others_batch=_flatten(T, N,value_preds_others_batch)
            
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)
            #print(share_obs_batch['rgb'].shape, rnn_states_batch.shape, actions_batch.shape, value_preds_batch.shape, return_batch.shape, masks_batch.shape, adv_targ.shape)
            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch



    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs={}
        obs={}
       
        #print("KEYs",self.share_obs.keys())

        if isinstance(self.share_obs, dict): 
            for key, value in self.share_obs.items():
                #print("shared observation shape",self.share_obs[key].shape,self.share_obs[key].transpose(1,2,0,3,4,5).shape)
                share_obs[key] =np.vstack( self.share_obs[key][:-1])#.transpose(1,2,0,3,4).reshape(-1, *self.share_obs[key].shape[2:])
                
                #print("shared observation shape", share_obs[key].shape)
                obs[key] =np.vstack(self.obs[key][:-1])#.transpose(1,2,0,3,4).reshape(-1, *self.obs[key].shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])
        
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            shared_obs_batch_dict=defaultdict(list)
            share_obs_batch_1=[]
            share_obs_batch_2=[]
            obs_batch_dict=defaultdict(list)
            obs_batch_1 = []
            obs_batch_2 = []
            
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                for key in self.share_obs.keys(): 
                    shared_obs_batch_dict[key].append(share_obs[key][ind:ind + data_chunk_length])
                    obs_batch_dict[key].append(obs[key][ind:ind + data_chunk_length])
              
                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            for key in self.share_obs.keys(): 
                shared_obs_batch_dict[key]=np.stack(shared_obs_batch_dict[key], 1)
                obs_batch_dict[key]=np.stack(obs_batch_dict[key], 1)

           

            actions_batch = np.stack(actions_batch,1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)
            value_preds_batch = np.stack(value_preds_batch,1)
            return_batch = np.stack(return_batch,1)
            masks_batch = np.stack(masks_batch,1)
            active_masks_batch = np.stack(active_masks_batch,1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch,1)
            adv_targ = np.stack(adv_targ,1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            for key in self.share_obs.keys(): 
                shared_obs_batch_dict[key]=_flatten(L, N, shared_obs_batch_dict[key])
                obs_batch_dict[key]=_flatten(L, N, obs_batch_dict[key])

            
            #obs_batch = _flatten(L, N, obs_batch)
  

            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
          
            #print("Yielded batch shape",obs_batch_2.shape)

            '''
            f=plt.figure(1)
           
                #print(obs['panoramic_view'].shape)
            #print(image.shape)
            
           

            plt.imshow(obs_batch_dict['rgb'][0].transpose(1,2,0))            

            g= plt.figure(2)

            plt.imshow(shared_obs_batch_dict['rgb'][0].transpose(1,2,0))
            plt.pause(0.00001)
                #print("actions batch",self.actions[i][0]==actions[i][0],self.returns[i][0][0]==returns[i][0],self.masks[i][0][0]==masks[i][0])
                #print("actions",self.actions[i][0],actions[i][0])
            r=input("press button")'''

            yield shared_obs_batch_dict, obs_batch_dict, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

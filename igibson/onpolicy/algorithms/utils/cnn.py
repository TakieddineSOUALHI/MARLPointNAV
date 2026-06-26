import torch.nn as nn
from .util import init
import torch
import matplotlib.pyplot as plt 
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from gym import spaces


def get_flattened_obs_dim(observation_space) :
        return spaces.utils.flatdim(observation_space)

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning.
    """

    def __init__(self, obs_shape, hidden_size,observation_space, use_orthogonal, use_ReLU,centralized_v=False,key='rgb'):
        super(NatureCNN, self).__init__()
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
   
        self.cnn=nn.Sequential(
            nn.Conv2d(obs_shape, 32, kernel_size=7, stride=2, padding=0),
            active_func,
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            active_func,
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            active_func,
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            active_func,
            nn.AdaptiveAvgPool2d((1,1)),
        )
     
        with torch.no_grad():
            obs=observation_space[key].sample()
            n_flatten = self.cnn(torch.as_tensor(obs).unsqueeze(0).float()).shape[1]     
        
        hidden_size=int((hidden_size)-64)
        self.linear = nn.Sequential(nn.Linear(n_flatten, hidden_size), active_func)

    def forward(self, x):
        x_1 = self.cnn(x).flatten(1)
        return self.linear(x_1)



class Encoder(nn.Module):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self,args,obs_space ,centralized_v=False,cnn_output_dim = 256):
        super(Encoder, self).__init__()
        use_orthogonal = args.use_orthogonal
        extractors = {}
        hidden_size=args.hidden_size
        use_ReLU=args.use_ReLU

        for key, subspace in obs_space.spaces.items():

            if key=='task_obs' or key=='rgb' or key=='panoramic' :
                extractors[key] =NatureCNN(3, hidden_size,obs_space, use_orthogonal, use_ReLU,centralized_v,key=key) 
            if key=='rgbd' :
                extractors[key] =NatureCNN(4, hidden_size,obs_space, use_orthogonal, use_ReLU,centralized_v,key=key) 
            elif key=='depth': 
                extractors[key] =NatureCNN(1, hidden_size,obs_space, use_orthogonal, use_ReLU,centralized_v,key=key) 
            elif key=='task_obs_point': 
                extractors[key] = nn.Linear(get_flattened_obs_dim(subspace),64)
   
        self.extractors= nn.ModuleDict(extractors)

    def forward(self, observations):
        encoded_tensor_list = []
        for key in observations.keys():
            if key!='proprioception':  
                encoded_tensor_list.append(self.extractors[key](observations[key]))

        return torch.cat(encoded_tensor_list, dim=1)



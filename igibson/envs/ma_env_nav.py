import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat

from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor

from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.image_nav_task import ImageNavTask
from igibson.tasks.ma_point_nav import MaPointNav
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW
import matplotlib.pyplot as plt 
import torch
import json
from torch import nn
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d
import math
import matplotlib.pyplot as plt 
from torchvision.transforms import ColorJitter
from igibson.sensors.dropout_sensor_noise import DropoutSensorNoise

log = logging.getLogger(__name__)



class Ma_Nav(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface).
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 1.0,
        physics_timestep=1 / 40.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        super(Ma_Nav, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            rendering_settings=rendering_settings,
            vr_settings=vr_settings,
            device_idx=device_idx,
            use_pb_gui=use_pb_gui,
        )
        self.automatic_reset = automatic_reset

        

    def load_task_setup(self):
        """
        Load task setup.
        """
        self.robots_num=self.config['num_agents']
      
        self.sensor_noise=False
        if self.config['sensor_noise']: 
            self.sensor_noise=True
            self.noise_model = DropoutSensorNoise(self)
            self.noise_model.set_noise_rate(0.99)

        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        

       
        # task
        if "task" not in self.config:
            self.task = DummyTask(self)
        elif self.config["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config["task"] == "image_nav":
            self.task = ImageNavTask(self)
        elif self.config["task"] == "ma_point_nav":
            self.task = MaPointNav(self)
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config["task"]))
            except ImportError:
                raise Exception("bddl is not available.")

    
    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    
  
    def load_observation_space(self):
        """
        Load observation space.
        """
        self.output = self.config["output"]
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
        observation_space = OrderedDict()
        share_observation_space=OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        
        
        if "task_obs_point" in self.output:
         
            observation_space["task_obs_point"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
            #share_observation_space["task_obs_point"] = self.build_obs_space(
            #    shape=(self.task.task_obs_dim*self.robots_num,), low=-np.inf, high=np.inf
            #)


        if "task_obs_pano" in self.output:
         
            observation_space["task_obs_pano"] = self.build_obs_space(
                shape=(3,self.image_height, self.image_width), low=0.0, high=1.0
            )

        if "task_obs_rgb" in self.output:
         
            observation_space["task_obs_rgb"] = self.build_obs_space(
                shape=(3,self.image_height, self.image_width), low=0.0, high=1.0
            )
            
            #share_observation_space["task_obs"]=self.build_obs_space(
            #    shape=(self.robots_num, 3,self.image_height, self.image_width*4), low=0.0, high=1.0
            #)
        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(3,self.image_height, self.image_width,), low=0.0, high=1.0
            )
            #share_observation_space["rgb"]=self.build_obs_space(
            #    shape=(6,self.image_height, self.image_width), low=0.0, high=1.0
            #)
            vision_modalities.append("rgb")
        
        if "panoramic" in self.output:
            observation_space["panoramic"] = self.build_obs_space(
                shape=(3,self.image_height, self.image_width*2,), low=0.0, high=1.0
            )
           
            #share_observation_space["panoramic"]=self.build_obs_space(
            #    shape=(6,self.image_height, self.image_width*2), low=0.0, high=1.0
            #)
        
            vision_modalities.append("panoramic")
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(1,self.image_height, self.image_width), low=0.0, high=1.0
            )
            vision_modalities.append("depth")

        if "rgbd" in self.output:
         
            observation_space["rgbd"] = self.build_obs_space(
                shape=(4,self.image_height, self.image_width,), low=0.0, high=1.0
            )

            vision_modalities.append("rgb")
            vision_modalities.append("depth")
        
        if "pc" in self.output:
            observation_space["pc"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("pc")
        if "optical_flow" in self.output:
            observation_space["optical_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
            )
            vision_modalities.append("optical_flow")
        if "scene_flow" in self.output:
            observation_space["scene_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("scene_flow")
        if "normal" in self.output:
            observation_space["normal"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("normal")
        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")
        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
            )
            vision_modalities.append("ins_seg")
        if "rgb_filled" in self.output:  # use filler
            observation_space["rgb_filled"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb_filled")
        if "highlight" in self.output:
            observation_space["highlight"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("highlight")
        if "scan" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "scan_rear" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan_rear"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan_rear")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")
        if "bump" in self.output:
            observation_space["bump"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
            sensors["bump"] = BumpSensor(self)
      
        if "proprioception" in self.output:
            observation_space["proprioception"] = self.build_obs_space(
                shape=(self.robots[0].proprioception_dim+3,), low=-np.inf, high=np.inf
            )
            share_observation_space["proprioception"]=self.build_obs_space(
                shape=((self.robots[0].proprioception_dim+3)*self.robots_num,), low=-np.inf, high=np.inf
            )
            sensors["bump"] = BumpSensor(self)

      
        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        if "scan_rear" in scan_modalities:
            sensors["scan_occ_rear"] = ScanSensor(self, scan_modalities, rear=True)

        
        self.observation_space = gym.spaces.Dict(observation_space)
        self.share_observation_space= gym.spaces.Dict(share_observation_space)
        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space.
        """   
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping.
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0#294#6140
        self.collision_links = []

    def load(self):
        """
        Load environment.
        """
        super(Ma_Nav, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self):
        """
        Get the current observation.

        :return: observation as a dictionary
        """
        state = OrderedDict()
        #if "rgbd" in self.output:
        #    state["rgb"] = self.task.get_task_obs(self)

        if "task_obs_point" in self.output:
            state["task_obs_point"] = self.task.get_task_obs(self)

        if "task_obs_rgb" in self.output:
            state["task_obs_rgb"] = self.task.get_task_obs(self)

        if "task_obs_pano" in self.output:
            state["task_obs_pano"] = self.task.get_task_obs(self)

        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            
            for modality in vision_obs:
                state[modality] = vision_obs[modality]

        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "scan_occ_rear" in self.sensors:
            scan_obs = self.sensors["scan_occ_rear"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
                
        if "proprioception" in self.output:
            target_pose = self.task.get_task_obs(self)  # (robots_num, 2) in robot-centric frame
            proporioception_data=torch.from_numpy(np.stack([self.robots[i].get_proprioception() for i in range(self.robots_num)]))
            collisions=torch.from_numpy(self.sensors["bump"].get_obs(self))
            state["proprioception"] = torch.cat((proporioception_data,target_pose,collisions.unsqueeze(1)),dim=1)
            
        
        
        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
        """
        self.simulator_step()
        multi_agent_collision_links = [[] for _ in range(self.robots_num)]
        for i in range(self.robots_num):
            multi_agent_collision_links[i]= [
            collision for bid in self.robots[i].get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]
        return self.filter_collision_links(multi_agent_collision_links)

    def filter_collision_links(self, multi_agent_collision_links):
        """
        Filter out collisions that should be ignored.

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_multi_agent_collision_links = []
        for i, collision_links in enumerate(multi_agent_collision_links):
        # TODO: Improve this to accept multi-body robots.
            new_collision_links = []
            for item in collision_links:
            # ignore collision with body b
                if item[2] in self.collision_ignore_body_b_ids:
                    continue

            # ignore collision with robot link a
                if item[3] in self.collision_ignore_link_a_ids:
                    continue

            # ignore self collision with robot link a (body b is also robot itself)
                if item[2] == self.robots[i].base_link.body_id and item[4] in self.collision_ignore_link_a_ids:
                    continue
                new_collision_links.append(item)
            new_multi_agent_collision_links.append(new_collision_links)

        return new_multi_agent_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        :param info: the info dictionary to populate
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        for i in range(self.robots_num):
            self.robots[i].apply_action(action[i])

        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)
        state = self.get_state()
       
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)
        
        for key in state.keys() :
            if len(state[key].shape)>3 and isinstance(state[key],np.ndarray):
                state[key]=np.transpose(state[key],(0,3, 1,2))
            
            elif len(state[key].shape)>3 and isinstance(state[key],torch.Tensor):
            
                state[key]=state[key].permute((0,3, 1,2))
        

        for i in range(self.robots_num):
            if done[i] and self.automatic_reset:
                info["last_observation"] = state
                self.reset_variables()
                break
        
        
        return state, reward, done, info

    def check_collision(self, body_id, ignore_ids=[]):
        """
        Check whether the given body_id has collision after one simulator step

        :param body_id: pybullet body id
        :param ignore_ids: pybullet body ids to ignore collisions with
        :return: whether the given body_id has collision
        """
        self.simulator_step()
        collisions = [x for x in p.getContactPoints(bodyA=body_id) if x[2] not in ignore_ids]

        if log.isEnabledFor(logging.INFO):  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                log.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) > 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        
        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], pos[2] + offset])

    def test_valid_position(self, obj, pos, orn=None, ignore_self_collision=False):
        """
        Test if the robot or the object can be placed with no collision.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param ignore_self_collision: whether the object's self-collisions should be ignored.
        :return: whether the position is valid
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        ignore_ids = obj.get_body_ids() if ignore_self_collision else []
        has_collision = any(self.check_collision(body_id, ignore_ids) for body_id in obj.get_body_ids())
        return not has_collision

    def land(self, obj, pos, orn,):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if not land_success:
            log.warning("Object failed to land.")

        if is_robot:
            obj.reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links =[]
        for i in range(self.robots_num):
            self.collision_links.append([])

    

    def reset(self):
        """
        Reset episode.
        """
    
        self.task.reset(self)
        self.simulator.sync(force_sync=True)
        self.reset_variables()
        state = self.get_state()

        
        for key in state.keys(): 
            if len(state[key].shape)>3 and isinstance(state[key],np.ndarray):
                
                state[key]=np.transpose(state[key],(0,3, 1,2))
            
            elif len(state[key].shape)>3 and isinstance(state[key],torch.Tensor):
            
                state[key]=state[key].permute((0,3, 1,2))
       
        
        
        return state

    def reset_agent(self):
        """
        Reset the robot's joint configuration and base pose until no collision
        """
        self.initial_pos = np.array(self.reset_list[str(self.current_episode)]['init_pos'])
        self.initial_orn = np.array([[0.,0.,0.] for _ in range(self.robots_num)])
        self.target_pos = np.array(self.reset_list[str(self.current_episode)]['target_pos'])
        self.target_orn = np.array(self.reset_list[str(self.current_episode)]['target_orn'])

        for robot_id in range(self.robots_num):
            self.land(self.robots[robot_id], self.initial_pos[robot_id], self.initial_orn[robot_id])
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = Ma_Nav(config_file=args.config, mode=args.mode)
    
    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for _ in range(2000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()

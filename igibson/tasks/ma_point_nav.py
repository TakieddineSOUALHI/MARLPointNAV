import numpy as np
import pybullet as p
import math
from igibson.objects.visual_marker import VisualMarker
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.point_goal import PointGoal
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, perspective, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz
import matplotlib.pyplot as plt 
import torch
import json 
import os
import csv 


class MaPointNav(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(MaPointNav, self).__init__(env)
        self.reward_type = self.config.get("reward_type", "l2")
        self.termination_conditions = [
            Timeout(self.config),
            PointGoal(self.config),
           
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
        ]
        self.task_type= self.config.get("scenario", "specificgoal")
        #self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        #self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))
        #self.target_pos = np.array(self.config.get("target_pos", [5, 5, 0]))
        self.goal_format = self.config.get("goal_format", "polar")
        self.dist_tol = self.config.get("dist_tol", 0.5)

        self.visible_target = self.config.get("visible_target", False)
        self.visible_path = self.config.get("visible_path", False)
        self.floor_num = 0
        reset_file_root = self.config.get('reset_file_root', 'data/dataset')
        if not os.path.isabs(reset_file_root):
            reset_file_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), reset_file_root)
        self.reset_file_root = reset_file_root

        if self.config["training"]!='test':
            self.reset_file = os.path.join(self.reset_file_root, str(self.config["scenario"]),str(self.config["training"]),str(self.config["num_agents"]), self.config["scene_id"]+'.json')
        else:
            self.reset_file = os.path.join(self.reset_file_root, str(self.config["scenario"]),str(self.config["training"]),str(self.config["num_agents"]),str(self.config["difficulty"]), self.config["scene_id"]+'.json')

        
        self.fieldnames=[]
        for i in range(int(self.config["num_agents"])): 
            self.fieldnames.append('agent_'+str(i))
        self.fieldnames.append('episode')
        self.fieldnames.append('floor_num')
        self.counter=0
        
        with open(self.reset_file, 'r') as f:
            self.reset_list = json.load(f)

        self.load_visualization(env)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        if env.mode != "gui_interactive":
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj =[]
        self.target_pos_vis_obj=[]
        for i in range(env.robots_num): 

            self.initial_pos_vis_obj.append(VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        ))
            self.target_pos_vis_obj.append(VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        ))

            env.simulator.import_object(self.initial_pos_vis_obj[i])
            env.simulator.import_object(self.target_pos_vis_obj[i])

        # The visual object indicating the initial location is always hidden
            for instance in self.initial_pos_vis_obj[i].renderer_instances:
                instance.hidden = True

        # The visual object indicating the target location may be visible
            for instance in self.target_pos_vis_obj[i].renderer_instances:
                instance.hidden = not self.visible_target

        if env.scene.build_graph:
            self.num_waypoints_vis = 10
            self.waypoints_vis = [
                VisualMarker(
                    visual_shape=p.GEOM_CYLINDER,
                    rgba_color=[0, 1, 0, 0.3],
                    radius=0.1,
                    length=cyl_length,
                    initial_offset=[0, 0, cyl_length / 2.0],
                )
                for _ in range(self.num_waypoints_vis)
            ]
            for waypoint in self.waypoints_vis:
                env.simulator.import_object(waypoint)
                # The path to the target may be visible
                for instance in waypoint.renderer_instances:
                    instance.hidden = not self.visible_path

    def get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """

        short_paths = self.get_shortest_path(env)
       
        geodesic_dist = [short_path[1] for short_path in short_paths]  # for multi agent, it's a list

        return geodesic_dist

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        l2_potential=[l2_distance(env.robots[i].get_position()[:2], self.target_pos[i][:2]) for i in range(env.robots_num)]
        return l2_potential

    def get_l2_distance(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        l2_potential=[l2_distance(env.robots[i].get_position()[:2], self.target_pos[i][:2]) for i in range(env.robots_num)]
        return l2_potential

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        if self.reward_type == "l2":
            return self.get_l2_potential(env)
        elif self.reward_type == "geodesic":
            return self.get_geodesic_potential(env)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        
        current_episode= env.current_episode
  
        self.initial_pos = np.array(self.reset_list[str(current_episode)]['init_pos'])
        self.initial_orn = np.array([[0.,0.,0.] for _ in range(env.robots_num)])
        self.target_pos = np.array(self.reset_list[str(current_episode)]['target_pos'])
        self.target_orn = np.array(self.reset_list[str(current_episode)]['target_orn'])
        self.floor=np.array(self.reset_list[str(current_episode)]['floor_num'])
      

    def reset_variables(self, env):
        
        self.path_length = [0 for i in range(env.robots_num)]
        self.robot_pos = [self.initial_pos[i][:2] for i in range(env.robots_num)]
        self.geodesic_dist = self.get_geodesic_potential(env)
        
        


    
    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(MaPointNav, self).get_termination(env, collision_links, action, info)
        geodeic_dist=self.get_geodesic_potential(env)
        for i in range(env.robots_num): 
            info["path_length"+str(i)] = self.path_length[i]
            if done[i]:
                info["spl"+str(i)] = (self.geodesic_dist[i])/max(self.geodesic_dist[i] ,self.path_length[i])
            else:
                info["spl"+str(i)] = 0.0
            info["dts"+str(i)]=geodeic_dist[i]
        return done, info

    def global_to_local(self, env, pos,j):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(np.array(pos) - np.array(env.robots[j].get_position()), *env.robots[j].get_rpy())

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """

        
        task_obs=[]
        for i in range(env.robots_num): 
            if self.task_type=='specificgoal':
                task_obs_bot=self.global_to_local(env, self.target_pos[i],i)[:2]

            if self.task_type=='commongoal' or 'adhoc': 
                task_obs_bot=self.global_to_local(env, self.target_pos[0],i)[:2]
            if self.goal_format == "polar":
                task_obs_bot = np.array(cartesian_to_polar(task_obs_bot[0], task_obs_bot[1]))

            task_obs.append(task_obs_bot)

        #plt.pause(0.0001)
        return torch.from_numpy(np.asarray(task_obs))
    
    
    def get_shortest_path(self,env, from_initial_pos=False, entire_path=False):
        """
        :param from_initial_pos: whether source is initial positions rather than current positions for multi agents
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position - multi agent,  is a list []
        """
        if from_initial_pos:
            sources = [init_pos[:2] for init_pos in self.initial_pos]
        else:
            sources = [robot.get_position()[:2] for robot in env.robots]
        if self.task_type == 'commongoal' or 'adhoc':
            short_paths = [env.scene.get_shortest_path(self.floor_num, sources[i], self.target_pos[0][:2], entire_path=entire_path) for i in range(env.robots_num)]
        elif self.task_type == 'specificgoal':
            short_paths = [env.scene.get_shortest_path(self.floor_num, sources[i], self.target_pos[i][:2], entire_path=entire_path) for i in range(env.robots_num)]
        return short_paths
    
    
    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != "gui_interactive":
            return

        for i in range(env.robots_num):
            self.initial_pos_vis_obj[i].set_position(self.initial_pos[i])
            if self.task_type=='commongoal' or 'adhoc':
                self.target_pos_vis_obj[i].set_position(self.target_pos[0])
            
            if self.task_type=='specificgoal':
                self.target_pos_vis_obj[i].set_position(self.target_pos[i])


        
        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path[0].shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([shortest_path[0][i][0], shortest_path[0][i][1], floor_height])
                )
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 100.0]))

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        multi_l2_distance=[]
        self.step_visualization(env)
        new_robot_pos = [env.robots[i].get_position()[:2] for i in range(env.robots_num)]#env.robots[0].get_position()[:2]
        for i in range(env.robots_num):
            self.path_length[i] += l2_distance(self.robot_pos[i], new_robot_pos[i])
        
       
        self.robot_pos = new_robot_pos
     
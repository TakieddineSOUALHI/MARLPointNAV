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

class ImageNavTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(ImageNavTask, self).__init__(env)
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
        self.task_type= self.config.get("task_type", "specific")
        #self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        #self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))
        #self.target_pos = np.array(self.config.get("target_pos", [5, 5, 0]))
        self.goal_format = self.config.get("goal_format", "polar")
        self.dist_tol = self.config.get("dist_tol", 0.5)
        sensors=self.config.get("output")
        #if "rgb" in sensors : 
        self.perspective_sensor=True
        self.visible_target = self.config.get("visible_target", False)
        self.visible_path = self.config.get("visible_path", False)
        self.floor_num = 0
        reset_file_root = self.config.get('reset_file_root', 'data/dataset')
        if not os.path.isabs(reset_file_root):
            reset_file_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), reset_file_root)
        self.reset_file_root = reset_file_root

        self.reset_file = os.path.join(self.reset_file_root, str(self.config["scenario"]), str(self.config["training"]),str(self.config["num_agents"]), self.config["scene_id"]+'.json')
        print("RESET FILE",self.reset_file)
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
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )
        self.target_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        env.simulator.import_object(self.initial_pos_vis_obj)
        env.simulator.import_object(self.target_pos_vis_obj)

        # The visual object indicating the initial location is always hidden
        for instance in self.initial_pos_vis_obj.renderer_instances:
            instance.hidden = True

        # The visual object indicating the target location may be visible
        for instance in self.target_pos_vis_obj.renderer_instances:
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
        #print("Target positions",self.initial_pos[0])

        #print("Calculated sor paths",short_paths[0])
        #print("Target positions",self.target_pos[0])
        geodesic_dist = [short_path[1] for short_path in short_paths]  # for multi agent, it's a list
        #print("Geodisic dist",geodesic_dist)

        return geodesic_dist

    def get_l2_potential(self, env):
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
        current_episode=env.current_episode
        index=0#np.random.randint(low=0, high=9999)
        #print("Current Episode index",index)
        
        self.initial_pos = np.array(self.reset_list[str(current_episode)]['init_pos'])
        self.initial_orn = np.array([[0.,0.,0.] for _ in range(env.robots_num)])
        self.target_pos = np.array(self.reset_list[str(current_episode)]['target_pos'])
        self.target_orn = np.array(self.reset_list[str(current_episode)]['target_orn'])
        ''''
        init_pos=[]
        init_ori=[]
        for i in range(env.robots_num):
            flag=True
            while flag==True: 
                a=np.random.normal(loc=0,scale=2)
                b=np.random.normal(loc=5.5,scale=0.5)            
                if (b-(abs(2*a)+8))<0:
                    flag= False

                    self.x=b
                    self.y=a
      
            self.yaw=-math.atan(self.y/(10-self.x))
            noise=np.random.normal(loc=0,scale=0.15)
            self.yaw+=noise
            init_pos.append([self.x,self.y,0.04])
            init_ori.append([0,0,self.yaw])

        self.initial_pos = np.array(init_pos)
        self.initial_orn = np.array(init_ori)
        self.target_pos = np.array([[8.,0.,0.] for _ in range(env.robots_num)])
        self.target_orn =  np.array([[0.,0.,0.] for _ in range(env.robots_num)])'''
        #print("Reset List",self.initial_pos,self.target_pos)
        #self.initial_pos[0]=[-2.2,4.3,-2.66000128]
        #self.target_pos[0]=[-4.6    ,     2.6    ,    -2.66000128]
      
        #print(self.initial_pos[0],self.target_pos[0],self.initial_pos [1],self.target_pos[1])
        
        for i in range(env.robots_num): 
            env.land(env.robots[i], self.initial_pos[i], self.initial_orn[i])

    def reset_variables(self, env):
        
        self.path_length = [0 for i in range(env.robots_num)]
        self.robot_pos = [self.initial_pos[i][:2] for i in range(env.robots_num)]
        self.geodesic_dist = self.get_geodesic_potential(env)
        raw_target_image=[]
        hidden_instances=[]
        for i in range(env.robots_num): 
            hidden_instances+=env.robots[i].renderer_instances


        for i in range(env.robots_num):
            camera_pos = [self.target_pos[i][0],self.target_pos[i][1],env.robots[i].eyes.get_position()[2]]
            orn=p.getQuaternionFromEuler(self.target_orn[i])
            
            mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
            view_direction = mat.dot(np.array([1, 0, 0]))
            up_direction = mat.dot(np.array([0, 0, 1]))
            env.simulator.renderer.set_camera(camera_pos, camera_pos + view_direction, up_direction)
            if self.perspective_sensor: 
                raw_target_image.append(env.simulator.renderer.render(modes=("rgb"),hidden=hidden_instances)[0])
            else :
                raw_target_image.append(env.simulator.renderer.get_equi())

        if env.mode == 'headless_tensor': 
            #print("Generated target_image",raw_target_image[0].shape)

            self.target_images=torch.stack([rgb[:, :, :3] for rgb in raw_target_image])
        else : 
            self.target_images=torch.from_numpy(np.stack([rgb[:, :, :3] for rgb in raw_target_image]))
        

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(ImageNavTask, self).get_termination(env, collision_links, action, info)
        
        for i in range(env.robots_num): 
            #print(i)
            info["path_length"+str(i)] = self.path_length[i]
            if done[i]:
                info["spl"+str(i)] = float(info["success"+str(i)]) * min(1.0, self.geodesic_dist[i] / self.path_length[i])
            else:
                info["spl"+str(i)] = 0.0

        return done, info

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(np.array(pos) - np.array(env.robots[0].get_position()), *env.robots[0].get_rpy())

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        
        
        
     


        

        '''
        task_obs = self.global_to_local(env, self.target_pos[0])[:2]
        if self.goal_format == "polar":
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(task_obs, [linear_velocity, angular_velocity])'''
        #plt.imshow(self.target_images[1].astype('uint8'))
        #plt.pause(0.0001)
        return self.target_images

    
    
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
        if self.task_type == 'common':
            short_paths = [env.scene.get_shortest_path(self.floor_num, sources[i], self.target_pos[0][:2], entire_path=entire_path) for i in range(env.robots_num)]
        elif self.task_type == 'specific':
            short_paths = [env.scene.get_shortest_path(self.floor_num, sources[i], self.target_pos[i][:2], entire_path=entire_path) for i in range(env.robots_num)]
        return short_paths
    
    
    
    
    '''
    
    def get_shortest_path(self, env, from_initial_pos=False, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[0][:2]
        print("Target and Pose",self.target_pos[0],source)
        return env.scene.get_shortest_path(self.floor_num, source, target, entire_path=entire_path)'''

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != "gui_interactive":
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos[0])
        self.target_pos_vis_obj.set_position(self.target_pos[0])

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
        #print("positions",self.robot_pos, new_robot_pos)
        for i in range(env.robots_num):
            #print("L2 distance return",l2_distance(self.robot_pos, new_robot_pos))
            self.path_length[i] += l2_distance(self.robot_pos[i], new_robot_pos[i])
       
        self.robot_pos = new_robot_pos

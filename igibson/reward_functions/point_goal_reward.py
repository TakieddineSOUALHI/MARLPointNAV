from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class PointGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(PointGoalReward, self).__init__(config)
        self.success_reward = self.config.get("success_reward", 10.0)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        reward=[]
        for i in range(env.robots_num): 
            if task.task_type=='commongoal' or task.task_type=='adhoc':
                success = l2_distance(env.robots[i].get_position()[:2], task.target_pos[0][:2]) < self.dist_tol
            if task.task_type=='specificgoal': 
                success = l2_distance(env.robots[i].get_position()[:2], task.target_pos[i][:2]) < self.dist_tol


            rew = self.success_reward if success else 0.0
            reward.append(rew)
        
        if self.shared_reward: 
            reward=[sum(reward)] * len(reward)
        
        return reward

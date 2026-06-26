from igibson.reward_functions.reward_function_base import BaseRewardFunction


class CollisionReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(CollisionReward, self).__init__(config)
        self.collision_reward_weight = self.config.get("collision_reward_weight", -0.05)
        

    def get_reward(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """ 
        reward=[]
        for i in range(env.robots_num): 

            has_collision = float(len(env.collision_links[i]) > 0)
            reward.append(has_collision * self.collision_reward_weight)
        
        if self.shared_reward: 
            reward=[sum(reward)] * len(reward)
        return reward

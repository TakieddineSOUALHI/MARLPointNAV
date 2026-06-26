from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition
from igibson.utils.utils import l2_distance


class PointGoal(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        dones=[]
        successes=[]

        for i in range(env.robots_num):
            if task.task_type=='commongoal' or task.task_type=='adhoc':
                dones.append(l2_distance(env.robots[i].get_position()[:2], task.target_pos[0][:2]) < self.dist_tol)
                successes.append(l2_distance(env.robots[i].get_position()[:2], task.target_pos[0][:2]) < self.dist_tol)
            if task.task_type=='specificgoal':
                dones.append(l2_distance(env.robots[i].get_position()[:2], task.target_pos[i][:2]) < self.dist_tol)
                successes.append(l2_distance(env.robots[i].get_position()[:2], task.target_pos[i][:2]) < self.dist_tol)

        
        return dones, successes

from igibson.sensors.sensor_base import BaseSensor
import numpy as np

class BumpSensor(BaseSensor):
    """
    Bump sensor
    """

    def __init__(self, env):
        super(BumpSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get Bump sensor reading

        :return: Bump sensor reading
        """
        bump=[]
        for i in range(env.robots_num): 
            has_collision = float(len(env.collision_links[i]) > 0)
            bump.append(has_collision)
        return np.asarray(bump)

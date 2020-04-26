from pybulletgym.envs.roboschool.robots.locomotors import Walker2D
import numpy as np
from pybulletgym.envs import gym_utils as ObjectHelper

class WalkerDribble(Walker2D):
    def __init__(self):
        Walker2D.__init__(self)
        self.flag = None

    def robot_specific_reset(self, bullet_client):
        Walker2D.robot_specific_reset(self, bullet_client)
        self.reset_ball()
        # self.flag_reposition()

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def reset_ball(self):
        # print(self.parts)
        pos = self.parts["torso"].current_position()
        if self.flag:
            self._p.resetBasePositionAndOrientation(self.flag.bodies[0], [pos[0]+1, pos[1], 0.5], [0, 0, 0, 1])
        else:
            self.flag = ObjectHelper.get_ball(self._p, pos[0]+1, pos[1], 0.5)
        self.walk_target_x = 1000
        # self.walk_target_x = self.flag.current_position()[0]
        self.walk_target_y = self.flag.current_position()[1]
            # self.flag = ObjectHelper.get_ball(self._p, -1.5,0,0.05)
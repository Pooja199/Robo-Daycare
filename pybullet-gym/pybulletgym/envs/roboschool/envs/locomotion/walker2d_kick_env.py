from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors.walker_soccer import WalkerSoccer
import numpy as np
# from pybulletgym.envs.roboschool.robots.locomotors import Walker2D


class Walker2DKickBulletEnv(WalkerBaseBulletEnv):
    def __init__(self):
        self.robot = WalkerSoccer()
        # self.robot = Walker2D()
        WalkerBaseBulletEnv.__init__(self, self.robot)
        self.is_init = 0


    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        if self.is_init == 0:
            self.ball_previous_pos_x = self.robot.flag.current_position()[0]
            self.ball_previous_pos_y = self.robot.flag.current_position()[1]
            self.is_init = 1

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # self.alive = 0
        self.alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        self.alive = min(self.alive, 0)
        done = self.alive < 0

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        self.ball_bonus = 500*np.linalg.norm([self.robot.flag.current_position()[1] - self.ball_previous_pos_y, self.robot.flag.current_position()[0] - self.ball_previous_pos_x])
        self.ball_previous_pos_x = self.robot.flag.current_position()[0]
        self.ball_previous_pos_y = self.robot.flag.current_position()[1]

        potential_old = self.potential
        # self.curr_robot
        self.potential = self.robot.calc_potential()
        # self.progress = 1*float(self.potential - potential_old)
        self.progress = 0

        feet_collision_cost = 0.0
        for i, f in enumerate(self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("self.ball_bonus=")
            print(self.ball_bonus)
            print("self.alive=")
            print(self.alive)
            print("self.progress")
            print(self.progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            self.ball_bonus,
            self.alive,
            self.progress,
            # electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]

        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}


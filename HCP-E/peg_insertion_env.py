import os

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

from env_base import BaseEnv


class PegInsertionEnv(BaseEnv):
    def __init__(self,
                 robot_folders,
                 robot_dir,
                 substeps,
                 tol=0.02,
                 train=True,
                 with_kin=None,
                 with_dyn=None,
                 multi_goal=False,
                 ):
        super().__init__(robot_folders=robot_folders,
                         robot_dir=robot_dir,
                         substeps=substeps,
                         tol=tol,
                         train=train,
                         with_kin=with_kin,
                         with_dyn=with_dyn,
                         multi_goal=multi_goal)

    def reset(self, robot_id=None):
        if robot_id is None:
            self.robot_id = np.random.randint(0, self.train_robot_num, 1)[0]
        else:
            self.robot_id = robot_id
        self.reset_robot(self.robot_id)

        ob = self.get_obs()
        self.ep_reward = 0
        self.ep_len = 0
        return ob

    def step(self, action):
        scaled_action = self.scale_action(action[:self.act_dim])
        self.sim.data.ctrl[:self.act_dim] = scaled_action
        self.sim.step()
        ob = self.get_obs()
        peg_target = self.sim.data.get_site_xpos('target'),
        reward, dist, done = self.cal_reward(ob[1].copy(),
                                             peg_target,
                                             action)

        self.ep_reward += reward
        self.ep_len += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'reward': reward, 'step': 1, 'dist': dist}
        return ob, reward, done, info

    def gen_random_delta_xyz(self):
        delta_xyz_range = np.array([[-0.20, 0.20],
                                    [-0.20, 0.20],
                                    [-0.20, 0.20]])

        starts = delta_xyz_range[:, 0]
        widths = delta_xyz_range[:, 1] - delta_xyz_range[:, 0]
        ran_num = np.random.random(size=(delta_xyz_range.shape[0]))
        delta_xyz = starts + widths * ran_num
        return delta_xyz

    def reset_robot(self, robot_id):
        self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        robot_file = os.path.join(self.robots[robot_id], 'robot.xml')
        self.model = load_model_from_path(robot_file)
        self.sim = MjSim(self.model, nsubsteps=self.nsubsteps)
        self.update_action_space()
        if self.multi_goal:
            table_name = 'table'
            table_id = self.sim.model._body_name2id[table_name]
            self.sim.model.body_pos[table_id] += self.gen_random_delta_xyz()
        self.sim.reset()
        self.sim.step()
        if self.viewer is not None:
            self.viewer = MjViewer(self.sim)

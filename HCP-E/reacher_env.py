import json
import os

import numpy as np

from env_base import BaseEnv


class ReacherEnv(BaseEnv):
    def __init__(self,
                 robot_folders,
                 robot_dir,
                 substeps,
                 tol=0.02,
                 train=True,
                 with_kin=None,
                 with_dyn=None,
                 multi_goal=False):
        super().__init__(robot_folders=robot_folders,
                         robot_dir=robot_dir,
                         substeps=substeps,
                         tol=tol,
                         train=train,
                         with_kin=with_kin,
                         with_dyn=with_dyn,
                         multi_goal=multi_goal)

        pose_file = os.path.join(robot_dir, 'poses.json')
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        self.initial_poses = np.array(pose_data['initial_joint_angles'])

    def reset(self, initial_pos=None, goal=None, robot_id=None):
        if initial_pos is None or goal is None:
            initial_pos, goal = self.reset_start_goal()
        if robot_id is None:
            self.robot_id = np.random.randint(0, self.train_robot_num, 1)[0]
        else:
            self.robot_id = robot_id
        self.reset_robot(self.robot_id)

        for joint_id, joint in enumerate(self.joint_set):
            if joint in self.sim.model._site_name2id:
                actuator_id = self.sim.model._actuator_name2id[joint]
                self.sim.data.qpos[actuator_id] = initial_pos[joint_id]

        target_name = 'target'
        site_id = self.sim.model._site_name2id[target_name]
        self.sim.model.site_pos[site_id] = goal

        ob = self.get_obs()
        self.ep_reward = 0
        self.ep_len = 0
        return ob

    def step(self, action):
        scaled_action = self.scale_action(action[:self.act_dim])
        self.sim.data.ctrl[:self.act_dim] = scaled_action

        self.sim.step()
        ob = self.get_obs()
        re_target = self.sim.data.get_site_xpos('target')
        reward, dist, done = self.cal_reward(ob[1].copy(),
                                             re_target,
                                             action)

        self.ep_reward += reward
        self.ep_len += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'reward': reward, 'step': 1, 'dist': dist}
        return ob, reward, done, info

    def gen_random_goal(self):
        xyz_range = np.array([[0.3, 0.6],
                              [-0.3, 0.3],
                              [-0.1, 0.3]])

        starts = xyz_range[:, 0]
        widths = xyz_range[:, 1] - xyz_range[:, 0]
        xyz = starts + widths * np.random.random(size=1)
        return xyz

    def reset_start_goal(self):
        start_id = np.random.choice(self.initial_poses.shape[0], 1)[0]
        return self.initial_poses[start_id], self.gen_random_goal()

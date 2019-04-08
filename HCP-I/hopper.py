import json
import os

import mujoco_py
import numpy as np
from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer

from util import seeding


class HopperEnv:
    def __init__(self,
                 robot_folders,
                 robot_dir,
                 substeps,
                 train=True,
                 with_embed=None,
                 with_kin=None,
                 with_dyn=None,
                 ):
        self.with_embed = with_embed
        self.with_kin = with_kin
        self.with_dyn = with_dyn
        if self.with_kin:
            norm_file = 'stats/kin_stats.json'
            with open(norm_file, 'r') as f:
                stats = json.load(f)
            self.kin_mu = np.array(stats['mu']).reshape(-1)
            self.kin_simga = np.array(stats['sigma']).reshape(-1)

        if self.with_dyn:
            norm_file = 'stats/dyn_stats.json'
            with open(norm_file, 'r') as f:
                stats = json.load(f)
            self.dyn_mu = np.array(stats['mu']).reshape(-1)
            self.dyn_sigma = np.array(stats['sigma']).reshape(-1)
            self.dyn_min = np.array(stats['min']).reshape(-1)
            self.dyn_max = np.array(stats['max']).reshape(-1)

        self.metadata = {}
        self.viewer = None
        self.reward_range = (-np.inf, np.inf)
        self.nsubsteps = substeps
        self.spec = None
        self.seed()
        self.bodies = ['torso', 'thigh', 'leg', 'foot']
        self.links = ['torso_geom', 'thigh_geom', 'leg_geom', 'foot_geom']

        self.robots = []
        for folder in robot_folders:
            self.robots.append(os.path.join(robot_dir, folder))

        self.dir2id = {folder: idx for idx, folder in enumerate(self.robots)}
        self.robot_num = len(self.robots)
        self.robot_id = 0
        self.reset_robot(self.robot_id)

        if train:
            # in training, we used the last 100 robots
            # as the testing robots (validation)
            # and the first 100 robots as the train_test robots
            # these robots are used to generate the
            # learning curves in training time
            self.test_robot_num = min(100, self.robot_num)
            train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(train_robot_num,
                                             self.robot_num))
            self.train_test_robot_num = min(self.test_robot_num,
                                            train_robot_num)
            self.train_test_robot_ids = list(range(self.train_test_robot_num))
            self.train_test_conditions = self.train_test_robot_num
        else:
            self.test_robot_num = self.robot_num
            self.test_robot_ids = list(range(self.robot_num))

        self.test_conditions = self.test_robot_num
        self.train_robot_num = self.robot_num - self.test_robot_num \
            if not self.with_embed else self.robot_num
        print('Train robots: ', self.train_robot_num)
        print('Test robots: ', self.test_robot_num)
        bounds = self.sim.model.actuator_ctrlrange
        self.ctrl_low = bounds[:, 0]
        self.ctrl_high = bounds[:, 1]
        self.scaling = (self.ctrl_high - self.ctrl_low) * 0.5
        self.action_space = spaces.Box(self.ctrl_low,
                                       self.ctrl_high,
                                       dtype=np.float32)
        observation = self.get_obs()
        self.obs_dim = observation.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.ep_reward = 0
        self.ep_len = 0
        self.reset(robot_id=0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005,
                                                       high=.005,
                                                       size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005,
                                                       high=.005,
                                                       size=self.model.nv)
        self.set_state(qpos, qvel)
        return self.get_obs()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and\
               qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset_robot(self, robot_id):
        self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        robot_file = os.path.join(self.robots[robot_id], 'robot.xml')
        self.model = load_model_from_path(robot_file)
        self.sim = MjSim(self.model, nsubsteps=self.nsubsteps)
        self.sim.reset()
        if self.viewer is not None:
            self.viewer = MjViewer(self.sim)
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

    def reset(self, robot_id=None):
        if robot_id is None:
            self.robot_id = self.np_random.randint(0,
                                                   self.train_robot_num,
                                                   1)[0]
        else:
            self.robot_id = robot_id
        self.reset_robot(self.robot_id)
        ob = self.reset_model()
        self.ep_reward = 0
        self.ep_len = 0
        return ob

    def scale_action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def test_reset(self, cond):
        if cond >= len(self.test_robot_ids):
            return np.full((self.obs_dim), np.nan)
        robot_id = self.test_robot_ids[cond]
        return self.reset(robot_id=robot_id)

    def train_test_reset(self, cond):
        if cond >= len(self.train_test_robot_ids):
            return np.full((self.obs_dim), np.nan)
        robot_id = self.train_test_robot_ids[cond]
        return self.reset(robot_id=robot_id)

    def step(self, a):
        a = self.scale_action(a)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self.get_obs()
        self.ep_reward += reward
        self.ep_len += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'reward': reward, 'step': 1}
        return ob, reward, done, info

    @property
    def dt(self):
        return self.model.opt.timestep * self.nsubsteps

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def do_simulation(self, ctrl):
        self.sim.data.ctrl[:] = ctrl
        self.sim.step()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    def get_obs(self):
        ob = np.concatenate([self.sim.data.qpos.flat[1:],
                             np.clip(self.sim.data.qvel.flat, -10, 10)])
        if self.with_kin:
            link_length = self.get_link_length(self.sim)
            link_length = np.divide((link_length - self.kin_mu),
                                    self.kin_simga)
            ob = np.concatenate([ob, link_length])
        if self.with_dyn:
            body_mass = self.get_body_mass(self.sim)
            joint_friction = self.get_friction(self.sim)
            joint_damping = self.get_damping(self.sim)
            armature = self.get_armature(self.sim)
            dyn_vec = np.concatenate((body_mass, joint_friction,
                                      joint_damping, armature))
            # dyn_vec = np.divide((dyn_vec - self.dyn_mu), self.dyn_simga)
            dyn_vec = np.divide((dyn_vec - self.dyn_min),
                                self.dyn_max - self.dyn_min)
            ob = np.concatenate([ob, dyn_vec])
        if self.with_embed:
            ob = np.concatenate((ob, np.array([self.robot_folder_id])))
        return ob

    def get_link_length(self, sim):
        link_length = []
        for link in self.links:
            geom_id = sim.model._geom_name2id[link]
            link_length.append(sim.model.geom_size[geom_id][1])
        return np.asarray(link_length).reshape(-1)

    def get_damping(self, sim):
        return sim.model.dof_damping[3:].reshape(-1)

    def get_friction(self, sim):
        return sim.model.dof_frictionloss[3:].reshape(-1)

    def get_body_mass(self, sim):
        body_mass = []
        for body in self.bodies:
            body_id = sim.model._body_name2id[body]
            body_mass.append(sim.model.body_mass[body_id])
        return np.asarray(body_mass).reshape(-1)

    def get_armature(self, sim):
        return sim.model.dof_armature[3:].reshape(-1)

    def close(self):
        pass

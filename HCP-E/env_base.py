import json
import os

import numpy as np
from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer

from util import rotations


class BaseEnv:
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
        self.with_kin = with_kin
        self.with_dyn = with_dyn
        self.multi_goal = multi_goal
        self.goal_dim = 3

        if self.with_dyn:
            norm_file = os.path.join(robot_dir, 'stats/dyn_stats.json')
            with open(norm_file, 'r') as f:
                stats = json.load(f)
            self.dyn_mu = np.array(stats['mu']).reshape(-1)
            self.dyn_sigma = np.array(stats['sigma']).reshape(-1)
            self.dyn_min = np.array(stats['min']).reshape(-1)
            self.dyn_max = np.array(stats['max']).reshape(-1)

        self.nsubsteps = substeps
        self.metadata = {}
        self.reward_range = (-np.inf, np.inf)
        self.spec = None
        self.dist_tol = tol
        self.viewer = None

        self.links = ['gl0', 'gl1_1', 'gl1_2', 'gl2', 'gl3_1',
                      'gl3_2', 'gl4', 'gl5_1', 'gl5_2', 'gl6']
        self.bodies = ["connector_plate_base",
                       "electric_gripper_base",
                       "gripper_l_finger",
                       "gripper_l_finger_tip",
                       "gripper_r_finger",
                       "gripper_r_finger_tip",
                       "l0",
                       "l1",
                       "l2",
                       "l3",
                       "l4",
                       "l5",
                       "l6"]
        self.site_set = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        self.joint_set = self.site_set

        self.robots = []
        for folder in robot_folders:
            self.robots.append(os.path.join(robot_dir, folder))

        self.dir2id = {folder: idx for idx, folder in enumerate(self.robots)}
        self.robot_num = len(self.robots)

        if train:
            self.test_robot_num = min(100, self.robot_num)
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.train_robot_num,
                                             self.robot_num))
            self.train_test_robot_num = min(100,
                                            self.train_robot_num)
            self.train_test_robot_ids = list(range(self.train_test_robot_num))
            self.train_test_conditions = self.train_test_robot_num
        else:
            self.test_robot_num = self.robot_num
            self.train_robot_num = self.robot_num - self.test_robot_num
            self.test_robot_ids = list(range(self.robot_num))

        self.test_conditions = self.test_robot_num

        print('Train robots: ', self.train_robot_num)
        print('Test robots: ', self.test_robot_num)
        print('Multi goal:', self.multi_goal)
        self.reset_robot(0)

        self.ob_dim = self.get_obs()[0].size
        print('Ob dim:', self.ob_dim)

        high = np.inf * np.ones(self.ob_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.ep_reward = 0
        self.ep_len = 0

    def reset(self, robot_id=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def update_action_space(self):
        actuators = self.sim.model._actuator_name2id.keys()
        valid_joints = [ac for ac in actuators if ac in self.joint_set]
        self.act_dim = len(valid_joints)
        bounds = self.sim.model.actuator_ctrlrange[:self.act_dim]
        self.ctrl_low = np.copy(bounds[:, 0])
        self.ctrl_high = np.copy(bounds[:, 1])
        self.action_space = spaces.Box(self.ctrl_low, self.ctrl_high,
                                       dtype=np.float32)

    def scale_action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reset_robot(self, robot_id):
        self.robot_folder_id = self.dir2id[self.robots[robot_id]]
        robot_file = os.path.join(self.robots[robot_id], 'robot.xml')
        self.model = load_model_from_path(robot_file)
        self.sim = MjSim(self.model, nsubsteps=self.nsubsteps)
        self.update_action_space()
        self.sim.reset()
        self.sim.step()
        if self.viewer is not None:
            self.viewer = MjViewer(self.sim)

    def test_reset(self, cond):
        robot_id = self.test_robot_ids[cond]
        return self.reset(robot_id=robot_id)

    def train_test_reset(self, cond):
        robot_id = self.train_test_robot_ids[cond]
        return self.reset(robot_id=robot_id)

    def cal_reward(self, s, goal, a):
        dist = np.linalg.norm(s - goal)

        if dist < self.dist_tol:
            done = True
            reward_dist = 1
        else:
            done = False
            reward_dist = -1
        reward = reward_dist
        reward -= 0.1 * np.square(a).sum()
        return reward, dist, done

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    def get_obs(self):
        qpos = self.get_qpos(self.sim)
        qvel = self.get_qvel(self.sim)

        ob = np.concatenate([qpos, qvel])
        if self.with_kin:
            link_rela = self.get_xpos_xrot(self.sim)
            ob = np.concatenate([ob, link_rela])
        if self.with_dyn:
            body_mass = self.get_body_mass(self.sim)
            joint_friction = self.get_friction(self.sim)
            joint_damping = self.get_damping(self.sim)
            armature = self.get_armature(self.sim)
            dyn_vec = np.concatenate((body_mass, joint_friction,
                                      joint_damping, armature))
            dyn_vec = np.divide((dyn_vec - self.dyn_min),
                                self.dyn_max - self.dyn_min)
            ob = np.concatenate([ob, dyn_vec])
        target_pos = self.sim.data.get_site_xpos('target').flatten()
        ob = np.concatenate([ob, target_pos])
        achieved_goal = self.sim.data.get_site_xpos('ref_pt')
        return ob, achieved_goal

    def get_link_length(self, sim):
        link_length = []
        for link in self.links:
            geom_id = sim.model._geom_name2id[link]
            link_length.append(sim.model.geom_size[geom_id][1])
        return np.asarray(link_length).reshape(-1)

    def get_qpos(self, sim):
        angle_noise_range = 0.02
        qpos = sim.data.qpos[:self.act_dim]
        qpos += np.random.uniform(-angle_noise_range,
                                  angle_noise_range,
                                  self.act_dim)
        qpos = np.pad(qpos, (0, 7 - self.act_dim),
                      mode='constant', constant_values=0)
        return qpos.reshape(-1)

    def get_qvel(self, sim):
        velocity_noise_range = 0.02
        qvel = sim.data.qvel[:self.act_dim]
        qvel += np.random.uniform(-velocity_noise_range,
                                  velocity_noise_range,
                                  self.act_dim)
        qvel = np.pad(qvel, (0, 7 - self.act_dim),
                      mode='constant', constant_values=0)
        return qvel.reshape(-1)

    def get_xpos_xrot(self, sim):
        xpos = []
        xrot = []
        for joint_id in range(self.act_dim):
            joint = sim.model._actuator_id2name[joint_id]
            if joint == 'j0':
                pos1 = sim.data.get_body_xpos('base_link')
                mat1 = sim.data.get_body_xmat('base_link')
            else:
                prev_id = joint_id - 1
                prev_joint = sim.model._actuator_id2name[prev_id]
                pos1 = sim.data.get_site_xpos(prev_joint)
                mat1 = sim.data.get_site_xmat(prev_joint)
            pos2 = sim.data.get_site_xpos(joint)
            mat2 = sim.data.get_site_xmat(joint)
            relative_pos = pos2 - pos1
            rot_euler = self.relative_rotation(mat1, mat2)
            xpos.append(relative_pos)
            xrot.append(rot_euler)
        xpos = np.array(xpos).flatten()
        xrot = np.array(xrot).flatten()
        xpos = np.pad(xpos, (0, (7 - self.act_dim) * 3),
                      mode='constant', constant_values=0)
        xrot = np.pad(xrot, (0, (7 - self.act_dim) * 3),
                      mode='constant', constant_values=0)
        ref_pt_xpos = self.sim.data.get_site_xpos('ref_pt')
        ref_pt_xmat = self.sim.data.get_site_xmat('ref_pt')
        relative_pos = ref_pt_xpos - pos2
        rot_euler = self.relative_rotation(mat2, ref_pt_xmat)
        xpos = np.concatenate((xpos, relative_pos.flatten()))
        xrot = np.concatenate((xrot, rot_euler.flatten()))
        pos_rot = np.concatenate((xpos, xrot))
        return pos_rot

    def get_damping(self, sim):
        damping = sim.model.dof_damping[:self.act_dim]
        damping = np.pad(damping, (0, 7 - self.act_dim),
                         mode='constant', constant_values=0)
        return damping.reshape(-1)

    def get_friction(self, sim):
        friction = sim.model.dof_frictionloss[:self.act_dim]
        friction = np.pad(friction, (0, 7 - self.act_dim),
                          mode='constant', constant_values=0)
        return friction.reshape(-1)

    def get_body_mass(self, sim):
        body_mass = []
        for body in self.bodies:
            body_id = sim.model._body_name2id[body]
            body_mass.append(sim.model.body_mass[body_id])
        return np.asarray(body_mass).reshape(-1)

    def get_armature(self, sim):
        armature = sim.model.dof_armature[:self.act_dim]
        armature = np.pad(armature, (0, 7 - self.act_dim),
                          mode='constant', constant_values=0)
        return armature.reshape(-1)

    def relative_rotation(self, mat1, mat2):
        # return the euler x,y,z of the relative rotation
        # (w.r.t site1 coordinate system) from site2 to site1
        rela_mat = np.dot(np.linalg.inv(mat1), mat2)
        return rotations.mat2euler(rela_mat)

    def close(self):
        pass

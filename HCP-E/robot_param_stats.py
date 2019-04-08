import argparse
import json
import os
from datetime import datetime

import numpy as np
from mujoco_py import load_model_from_path, MjSim
from tqdm import tqdm

from util import rotations

joint_set = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']


def relative_rotation(mat1, mat2):
    # return the euler x,y,z of the relative rotation
    # (w.r.t site1 coordinate system) from site2 to site1
    rela_mat = np.dot(np.linalg.inv(mat1), mat2)
    return rotations.mat2euler(rela_mat)


def get_obs(sim, act_dim, type='kin'):
    if type == 'dyn':
        body_mass = get_body_mass(sim)
        joint_friction = get_friction(sim, act_dim)
        joint_damping = get_damping(sim, act_dim)
        armature = get_armature(sim, act_dim)
        dyn = np.concatenate(
            (body_mass, joint_friction, joint_damping, armature))
        return dyn
    elif type == 'kin':
        return get_link_length(sim)


def get_armature(sim, act_dim):
    armature = sim.model.dof_armature[:act_dim]
    armature = np.pad(armature, (0, 7 - act_dim),
                      mode='constant',
                      constant_values=0)
    return armature.reshape(-1)


def get_link_length(sim):
    links = ['gl0', 'gl1_1', 'gl1_2', 'gl2', 'gl3_1',
             'gl3_2', 'gl4', 'gl5_1', 'gl5_2', 'gl6']
    link_length = []
    for link in links:
        geom_id = sim.model._geom_name2id[link]
        link_length.append(sim.model.geom_size[geom_id][1])
    return np.asarray(link_length).reshape(-1)


def get_damping(sim, act_dim):
    damping = sim.model.dof_damping[:act_dim]
    damping = np.pad(damping, (0, 7 - act_dim),
                     mode='constant',
                     constant_values=0)
    return damping.reshape(-1)


def get_friction(sim, act_dim):
    friction = sim.model.dof_frictionloss[:act_dim]
    friction = np.pad(friction, (0, 7 - act_dim),
                      mode='constant',
                      constant_values=0)
    return friction.reshape(-1)


def get_body_mass(sim):
    bodies = ["connector_plate_base",
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
    body_mass = []
    for body in bodies:
        body_id = sim.model._body_name2id[body]
        body_mass.append(sim.model.body_mass[body_id])
    return np.asarray(body_mass).reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Statistics')
    parser.add_argument('--robot_dir',
                        default='../xml/gen_xmls/simrobot',
                        type=str,
                        help='path to robot configs')
    parser.add_argument('--save_dir', '-sd', type=str,
                        default='./stats',
                        help='directory to save triplet training data')
    parser.add_argument('--type', type=str,
                        default='dyn',
                        help='which kind of params to '
                             'get stats [\'kin\', \'dyn\']')
    print('Program starts at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    robot_folders = os.listdir(args.robot_dir)
    vecs = []
    for robot_folder in tqdm(robot_folders):
        robot_file = os.path.join(args.robot_dir,
                                  robot_folder,
                                  'robot.xml')
        model = load_model_from_path(robot_file)
        sim = MjSim(model, nsubsteps=5)
        actuators = sim.model._actuator_name2id.keys()
        valid_joints = [ac for ac in actuators if ac in joint_set]
        act_dim = len(valid_joints)
        vecs.append(get_obs(sim, act_dim, type=args.type))
    vecs = np.array(vecs)
    mins = np.min(vecs, axis=0)
    maxs = np.max(vecs, axis=0)
    mus = np.mean(vecs, axis=0)
    sigmas = np.std(vecs, axis=0)
    print('Min: ', mins)
    print('Max: ', maxs)
    print('Mu: ', mus)
    print('Sigma: ', sigmas)
    stats = {'min': mins.tolist(),
             'max': maxs.tolist(),
             'mu': mus.tolist(),
             'sigma': sigmas.tolist()}
    with open(os.path.join(args.save_dir,
                           '%s_stats.json' % args.type),
              'w') as outfile:
        json.dump(stats, outfile, indent=4)
    print('Program ends at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))

import argparse
import json
import os
from datetime import datetime

import numpy as np
from mujoco_py import load_model_from_path, MjSim
from tqdm import tqdm


def get_obs(sim, type='kin'):
    if type == 'dyn':
        body_mass = get_body_mass(sim)
        joint_friction = get_friction(sim)
        joint_damping = get_damping(sim)
        armature = get_armature(sim)
        dyn = np.concatenate(
            (body_mass, joint_friction, joint_damping, armature))
        return dyn
    elif type == 'kin':
        return get_link_length(sim)


def get_link_length(sim):
    links = ['torso_geom', 'thigh_geom', 'leg_geom', 'foot_geom',
             'thigh_left_geom', 'leg_left_geom', 'foot_left_geom']
    link_length = []
    for link in links:
        if link not in sim.model._geom_name2id:
            continue
        geom_id = sim.model._geom_name2id[link]
        link_length.append(sim.model.geom_size[geom_id][1])
    return np.asarray(link_length).reshape(-1)


def get_armature(sim):
    return sim.model.dof_armature[3:].reshape(-1)


def get_damping(sim):
    return sim.model.dof_damping[3:].reshape(-1)


def get_friction(sim):
    return sim.model.dof_frictionloss[3:].reshape(-1)


def get_body_mass(sim):
    bodies = ['torso', 'thigh', 'leg', 'foot',
              'thigh_left', 'leg_left', 'foot_left']
    body_mass = []
    for body in bodies:
        if body not in sim.model._body_name2id:
            continue
        body_id = sim.model._body_name2id[body]
        body_mass.append(sim.model.body_mass[body_id])
    return np.asarray(body_mass).reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Statistics')
    parser.add_argument('--robot_dir',
                        default='./stats_robots',
                        type=str,
                        help='path to robot configs'
                             '(default: \'../robots/mjb_configs\')')
    parser.add_argument('--save_dir', '-sd', type=str, default='./stats',
                        help='directory to save triplet training data')
    parser.add_argument('--type', type=str, default='kin',
                        help='which kind of params to '
                             'get stats [\'kin\', \'dyn\']')
    print('Program starts at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    robot_files = os.listdir(args.robot_dir)
    embed_vecs = []
    for robot_file in tqdm(robot_files):
        robot_file = os.path.join(args.robot_dir,
                                  robot_file,
                                  'robot.xml')
        model = load_model_from_path(robot_file)
        sim = MjSim(model, nsubsteps=5)
        embed_vecs.append(get_obs(sim, type=args.type))
    embed_vecs = np.array(embed_vecs)
    mins = np.min(embed_vecs, axis=0)
    maxs = np.max(embed_vecs, axis=0)
    mus = np.mean(embed_vecs, axis=0)
    sigmas = np.std(embed_vecs, axis=0)
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

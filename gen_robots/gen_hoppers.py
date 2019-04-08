import argparse
import copy
import os
import time

import numpy as np
from mujoco_py import load_model_from_path, MjSim
from scipy.stats import truncnorm
from tqdm import tqdm


def truncated_norm(mu, sigma, lower_limit, upper_limit, size):
    if sigma == 0:
        return mu
    lower_limit = (lower_limit - mu) / sigma
    upper_limit = (upper_limit - mu) / sigma
    r = truncnorm.rvs(lower_limit, upper_limit, size=size)
    return r * sigma + mu


def var_one_link_len(sim_env, vars_dict, link_name):
    tmp_vars_dict = copy.deepcopy(vars_dict)
    link_len = tmp_vars_dict["link_length"]
    for link, param in link_len.items():
        if link == link_name:
            continue
        else:
            param[1] = 0
    return robot_param_random_sample(sim_env, tmp_vars_dict)


def angle_axis_to_quaternion(x, y, z, angle):
    qx = x * np.sin(angle / 2.0)
    qy = y * np.sin(angle / 2.0)
    qz = z * np.sin(angle / 2.0)
    qw = np.cos(angle / 2.0)
    return np.ravel(np.array([qw, qx, qy, qz]))


def robot_param_random_sample(sim_env, vars_to_change, pre_gen_params):
    vars_dict = get_variable_params()
    if vars_to_change.get('link_length', False):
        link_lens = vars_dict["link_length"]
        pre_param_id = np.random.randint(0,
                                         len(pre_gen_params['link_length']),
                                         1)[0]
        for link, param in link_lens.items():
            geom_id = sim_env.model._geom_name2id[link]
            llen = pre_gen_params['link_length'][pre_param_id][link]
            half_llen = llen / 2.0
            delta_llen = llen - sim_env.model.geom_size[geom_id][1] * 2
            if link in ['torso_geom', 'thigh_geom', 'leg_geom']:
                sim_env.model.geom_size[geom_id][1] = half_llen
                sim_env.model.geom_pos[geom_id][2] = half_llen
            else:
                sim_env.model.geom_size[geom_id][1] = half_llen
                sim_env.model.geom_pos[geom_id][0] = -half_llen

            if link == 'thigh_geom':
                body_id = sim_env.model._body_name2id['thigh']
                sim_env.model.body_pos[body_id][2] -= delta_llen
                jnt_id = sim_env.model._joint_name2id['thigh_joint']
                sim_env.model.jnt_pos[jnt_id][2] += delta_llen
            elif link == 'leg_geom':
                body_id = sim_env.model._body_name2id['leg']
                sim_env.model.body_pos[body_id][2] -= delta_llen
                jnt_id = sim_env.model._joint_name2id['leg_joint']
                sim_env.model.jnt_pos[jnt_id][2] += delta_llen
            elif link == 'foot_geom':
                body_id = sim_env.model._body_name2id['foot']
                sim_env.model.body_pos[body_id][0] += delta_llen
                jnt_id = sim_env.model._joint_name2id['foot_joint']
                sim_env.model.jnt_pos[jnt_id][0] -= delta_llen
            elif link == 'torso_geom':
                jnt_id = sim_env.model._joint_name2id['rootx']
                sim_env.model.jnt_pos[jnt_id] = np.zeros(3)
                jnt_id = sim_env.model._joint_name2id['rootz']
                sim_env.model.jnt_pos[jnt_id] = np.zeros(3)
                jnt_id = sim_env.model._joint_name2id['rooty']
                sim_env.model.jnt_pos[jnt_id][2] += delta_llen / 2.0
        sim_env.step()
        foot_id = sim_env.model._body_name2id['leg']
        z_coord = sim_env.data.body_xpos[foot_id][2] - 0.1
        torso_id = sim_env.model._body_name2id['torso']
        sim_env.model.body_pos[torso_id][2] -= z_coord

    if vars_to_change.get('damping', False):
        for idx in range(3, 6):
            pre_param_id = np.random.randint(0,
                                             len(pre_gen_params['damping']),
                                             1)[0]
            vdamp = pre_gen_params['damping'][pre_param_id]
            sim_env.model.dof_damping[idx] = vdamp

    if vars_to_change.get('friction', False):
        for idx in range(3, 6):
            pre_param_id = np.random.randint(0,
                                             len(pre_gen_params['friction']),
                                             1)[0]
            vfriction = pre_gen_params['friction'][pre_param_id]
            sim_env.model.dof_frictionloss[idx] = vfriction

    if vars_to_change.get('body_mass', False):
        body_mass = vars_dict["body_mass"]
        for body in body_mass:
            pre_param_id = np.random.randint(0,
                                             len(pre_gen_params['body_mass']),
                                             1)[0]
            body_id = sim_env.model._body_name2id[body]
            vbmass = pre_gen_params['body_mass'][pre_param_id]
            sim_env.model.body_mass[body_id] *= vbmass

    if vars_to_change.get('armature', False):
        for idx in range(3, 6):
            pre_param_id = np.random.randint(0,
                                             len(pre_gen_params['armature']),
                                             1)[0]
            varmature = pre_gen_params['armature'][pre_param_id]
            sim_env.model.dof_armature[idx] *= varmature


def pre_gen_robot_params(vars_to_change, param_var_num):
    vars_dict = get_variable_params()
    pre_gen_params = {}
    if vars_to_change.get('link_length', False):
        link_lens = vars_dict["link_length"]
        pre_gen_params['link_length'] = {}
        for iter in range(param_var_num):
            pre_gen_params['link_length'][iter] = {}
            for link, param in link_lens.items():
                llen = np.random.uniform(param[0] - param[1],
                                         param[0] + param[1],
                                         1)
                pre_gen_params['link_length'][iter][link] = llen

    if vars_to_change.get('damping', False):
        damping_range = vars_dict["damping_range"]
        if damping_range[1] <= 1 or damping_range[0] >= 1:
            pre_gen_params['damping'] = np.random.uniform(damping_range[0],
                                                          damping_range[1],
                                                          param_var_num)
        else:
            half_num = int(param_var_num / 2)
            under_damping = np.random.uniform(damping_range[0], 1, half_num)
            over_damping = np.random.uniform(1, damping_range[1], half_num)
            pre_gen_params['damping'] = np.concatenate((under_damping,
                                                        over_damping))

    if vars_to_change.get('friction', False):
        frictionloss_range = vars_dict["frictionloss_range"]
        pre_gen_params['friction'] = {}
        pre_gen_params['friction'] = np.random.uniform(frictionloss_range[0],
                                                       frictionloss_range[1],
                                                       param_var_num)

    if vars_to_change.get('body_mass', False):
        mass_ratio = vars_dict["mass_ratio"]
        if mass_ratio[1] <= 1 or mass_ratio[0] >= 1:
            pre_gen_params['body_mass'] = np.random.uniform(mass_ratio[0],
                                                            mass_ratio[1],
                                                            param_var_num)
        else:
            half_num = int(param_var_num / 2)
            under_one = np.random.uniform(mass_ratio[0], 1, half_num)
            over_one = np.random.uniform(1, mass_ratio[1], half_num)
            pre_gen_params['body_mass'] = np.concatenate((under_one, over_one))

    if vars_to_change.get('armature', False):
        armature_ratio = vars_dict["armature_ratio"]
        if armature_ratio[1] <= 1 or armature_ratio[0] >= 1:
            pre_gen_params['armature'] = np.random.uniform(armature_ratio[0],
                                                           armature_ratio[1],
                                                           param_var_num)
        else:
            half_num = int(param_var_num / 2)
            under_one = np.random.uniform(armature_ratio[0], 1, half_num)
            over_one = np.random.uniform(1, armature_ratio[1], half_num)
            pre_gen_params['armature'] = np.concatenate((under_one, over_one))
    return pre_gen_params


def get_variable_params():
    '''
    link_length: [mu, sigma]
    damping_range: [lower_bound, upper_bound]
    frictionloss_range: [lower_bound, upper_bound]
    mass_ratio: [lower_bound_ratio, upper_bound_ratio]
    armature_ratio: [lower_bound_ratio, upper_bound_ratio]
    body_mass: bodies whose mass needs to be changed
    '''
    vars_dict = {"link_length": {"torso_geom": [0.4, 0.1],
                                 "thigh_geom": [0.45, 0.1],
                                 "leg_geom": [0.5, 0.15],
                                 "foot_geom": [0.39, 0.1]},
                 "damping_range": [0.01, 5.0],
                 "frictionloss_range": [0, 2],
                 "mass_ratio": [0.25, 2],
                 "armature_ratio": [0.1, 2],
                 "body_mass": ["torso",
                               "thigh",
                               "leg",
                               "foot"],
                 }
    return vars_dict


def gen_robot_configs(ref_file, robot_num, vars_to_change,
                      param_var_num, root_save_dir):
    pre_gen_params = pre_gen_robot_params(vars_to_change=vars_to_change,
                                          param_var_num=param_var_num)
    for idx in tqdm(range(robot_num)):
        model = load_model_from_path(ref_file)
        sim = MjSim(model)
        robot_param_random_sample(sim, vars_to_change, pre_gen_params)
        sim.reset()
        sub_save_dir = os.path.join(root_save_dir, 'robot_%d' % idx)
        os.makedirs(sub_save_dir, exist_ok=True)
        with open(os.path.join(sub_save_dir, 'robot.xml'), 'w') as f:
            sim.save(f, keep_inertials=True)


if __name__ == "__main__":
    desp = 'Generate n new robot configuration files randomly'
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('--robot_num', '-rn', type=int, default=1000,
                        help='number of robot configuration'
                             ' files to be generated')
    parser.add_argument('--param_var_num', '-pvn', type=int, default=10000,
                        help='size of the pool for each '
                             'kinematic and dynamic parameter')
    parser.add_argument('--random_seed', '-rs', type=int, default=1,
                        help='seed for random number')
    parser.add_argument('--ref_xml', '-rx', type=str,
                        default='../xml/hopper/hopper.xml',
                        help='reference robot xml file')
    parser.add_argument('--save_dir', '-sd', type=str,
                        default='../xml/gen_xmls/hopper',
                        help='save directory for generated xmls')
    args = parser.parse_args()
    np.random.seed(seed=args.random_seed)
    print('Reference file: ', os.path.abspath(args.ref_xml))
    assert os.path.exists(args.ref_xml)
    print('Generating files to: ', os.path.abspath(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    vars_to_change = {'link_length': True,
                      'damping': True,
                      'friction': True,
                      'body_mass': True,
                      'armature': True
                      }

    start = time.time()
    gen_robot_configs(ref_file=args.ref_xml,
                      robot_num=args.robot_num,
                      vars_to_change=vars_to_change,
                      param_var_num=args.param_var_num,
                      root_save_dir=args.save_dir)
    end = time.time()
    print("Generating %d robots took %.3f seconds"
          "" % (args.robot_num, end - start))

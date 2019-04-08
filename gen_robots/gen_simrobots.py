import argparse
import copy
import os
import re
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


def var_one_link_len(simEnv, varsDict, link_name):
    tmp_vars_dict = copy.deepcopy(varsDict)
    link_len = tmp_vars_dict["link_length"]
    for link, param in link_len.items():
        if link == link_name:
            continue
        else:
            param[1] = 0
    return robot_param_random_sample(simEnv, tmp_vars_dict)


def angle_axis_to_quaternion(x, y, z, angle):
    qx = x * np.sin(angle / 2.0)
    qy = y * np.sin(angle / 2.0)
    qz = z * np.sin(angle / 2.0)
    qw = np.cos(angle / 2.0)
    return np.ravel(np.array([qw, qx, qy, qz]))


def robot_param_random_sample(sim_env, vars_to_change, pre_gen_params,
                              valid_joints_num, skip_geom):
    vars_dict = get_variable_params()
    if vars_to_change.get('link_length', False):
        link_lens = vars_dict["link_length"]
        delta_llen = {}

        for link, param in link_lens.items():
            if link in skip_geom:
                delta_llen[link] = 0
                continue
            id = sim_env.model._geom_name2id[link]
            u_ran = len(pre_gen_params['link_length'])
            pre_param_id = np.random.randint(0,
                                             u_ran,
                                             1)[0]
            llen = pre_gen_params['link_length'][pre_param_id][link]
            half_llen = llen / 2.0
            delta_llen[link] = llen - sim_env.model.geom_size[id][1] * 2
            sim_env.model.geom_size[id][1] = half_llen
            sign = np.sign(sim_env.model.geom_pos[id][param[2]])
            sim_env.model.geom_pos[id][param[2]] = sign * half_llen
            if len(param) == 4:
                next_id = sim_env.model._geom_name2id[param[3]]
                add_r = delta_llen[link] * sign
                sim_env.model.geom_pos[next_id][param[2]] += add_r

        bodies = vars_dict["body_links"]
        for body, param in bodies.items():
            prev_links = param[0]
            pos_idices = param[1]
            id = sim_env.model._body_name2id[body]
            for en, pos_idx in enumerate(pos_idices):
                link = prev_links[en]
                sign = np.sign(sim_env.model.body_pos[id][pos_idx])
                sim_env.model.body_pos[id][pos_idx] += delta_llen[link] * sign

    if vars_to_change.get('link_radius', False):
        link_radius = vars_dict["link_radius"]
        for link, param in link_radius.items():
            mu = (param[0] + param[1]) / 2
            sigma = mu - param[0]
            r = truncated_norm(mu, sigma, param[0], param[1], 1)
            # r = np.random.uniform(param[0], param[1], 1)
            geom_id = sim_env.model._geom_name2id[link]
            delta_r = r - sim_env.model.geom_size[geom_id][0]
            sim_env.model.geom_size[geom_id][0] = r
            if len(param) == 3:
                for elink in param[2]:
                    geom_id = sim_env.model._geom_name2id[elink]
                    sim_env.model.geom_size[geom_id][0] += delta_r

    if vars_to_change.get('damping', False):
        for idx in range(valid_joints_num):
            pre_param_id = np.random.randint(0,
                                             len(pre_gen_params['damping']),
                                             1)[0]
            vdamp = pre_gen_params['damping'][pre_param_id]
            sim_env.model.dof_damping[idx] = vdamp

    if vars_to_change.get('friction', False):
        for idx in range(valid_joints_num):
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
        for idx in range(valid_joints_num):
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
            # critical damping
            half_num = int(param_var_num / 2)
            under_damping = np.random.uniform(damping_range[0], 1, half_num)
            over_damping = np.random.uniform(1, damping_range[1], half_num)
            pre_gen_params['damping'] = np.concatenate((under_damping,
                                                        over_damping))

    if vars_to_change.get('friction', False):
        frictionloss_range = vars_dict["frictionloss_range"]
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
    body_links: [[prev_links], [on which dimension
                 the body pos should be changed]]
    the two sublists should have same length
    link_length: [mu, sigma, index of pos,
                  the link whose pos will be
                  affected by this link (optional)]
    link_radius: [lower_bound, upper_bound,
                  [links whose radius will be determined by this link]]
    damping_range: [lower_bound, upper_bound]
    frictionloss_range: [lower_bound, upper_bound]
    mass_ratio: [lower_bound_ratio, upper_bound_ratio]
    body_mass: bodies whose mass needs to be changed
    camera_pos_range: position range in x, y, z
    camera_angle_range: camera rotation angle range w.r.t y axis
    table_height_range: lower_bound, upper_bound
    '''
    vars_dict = {"body_links": {"l1": [["gl0"], [2]],
                                "l2": [["gl1_1", "gl1_2"], [2, 1]],
                                "l3": [["gl2"], [2]],
                                "l4": [["gl3_1", "gl3_2"], [2, 1]],
                                "l5": [["gl4"], [2]],
                                "l6": [["gl5_1", "gl5_2"], [2, 1]],
                                "connector_plate_base": [["gl6"], [2]]},
                 "link_length": {"gl0": [0.29, 0.1, 2],
                                 "gl1_1": [0.119, 0.05, 2, "gl1_2"],
                                 "gl1_2": [0.140, 0.07, 1],
                                 "gl2": [0.263, 0.12, 2],
                                 "gl3_1": [0.120, 0.06, 2, "gl3_2"],
                                 "gl3_2": [0.127, 0.06, 1],
                                 "gl4": [0.275, 0.12, 2],
                                 "gl5_1": [0.0958, 0.04, 2, "gl5_2"],
                                 "gl5_2": [0.076, 0.03, 1],
                                 "gl6": [0.0488, 0.02, 2]},
                 # # 5dof_robot5_special:
                 # "link_length": {"gl0": [0.29, 0.1, 2],
                 #                 "gl1_1": [0.02, 0.0, 2, "gl1_2"],
                 #                 "gl1_2": [0.140, 0.07, 1],
                 #                 "gl2": [0.263, 0.12, 2],
                 #                 "gl3_1": [0.020, 0.0, 2, "gl3_2"],
                 #                 "gl3_2": [0.127, 0.06, 1],
                 #                 "gl4": [0.275, 0.12, 2],
                 #                 "gl5_1": [0.02, 0.0, 2, "gl5_2"],
                 #                 "gl5_2": [0.076, 0.03, 1],
                 #                 "gl6": [0.0488, 0.02, 2]},
                 "link_radius": {"gl1_1": [0.058, 0.078],
                                 "gl1_2": [0.043, 0.063, ["gl2", "gl3_1"]],
                                 "gl3_2": [0.035, 0.055, ["gl4"]],
                                 "gl5_1": [0.032, 0.052,
                                           ["gl5_2", "gl6",
                                            "gconnector_base"]]},
                 "damping_range": [0.01, 30],
                 "frictionloss_range": [0, 10],
                 "mass_ratio": [0.25, 4],
                 "armature_ratio": [0.01, 4],
                 "body_mass": ["connector_plate_base",
                               "connector_plate_mount",
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
                               "l6"],
                 }
    return vars_dict


def gen_robot_configs(ref_file, robot_num,
                      vars_to_change, param_var_num,
                      skip_geom, root_save_dir):
    pre_gen_params = pre_gen_robot_params(vars_to_change=vars_to_change,
                                          param_var_num=param_var_num)
    joint_sites = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    for idx in tqdm(range(robot_num)):
        model = load_model_from_path(ref_file)
        sim = MjSim(model)
        valid_joint_sites = [el for el in joint_sites
                             if el in sim.model._site_name2id]
        valid_joints_num = len(valid_joint_sites)

        robot_param_random_sample(sim,
                                  vars_to_change,
                                  pre_gen_params,
                                  valid_joints_num,
                                  skip_geom)
        sim.model.vis.map.znear = 0.02
        sim.model.vis.map.zfar = 50.0
        sim.reset()
        robot_id = int(re.findall(r'\d+', ref_file.split('/')[-1])[0])
        save_dir = os.path.join(root_save_dir, '%d_dof_%d_%d' %
                                (valid_joints_num, robot_id, idx))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'robot.xml'), 'w') as f:
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
                        default='../xml/simrobot/7dof/peg/robot1.xml',
                        help='reference robot xml file')
    parser.add_argument('--save_dir', '-sd', type=str,
                        default='../xml/gen_xmls/simrobot',
                        help='save directory for generated xmls')
    args = parser.parse_args()
    np.random.seed(seed=args.random_seed)
    print('Reference file: ', os.path.abspath(args.ref_xml))
    assert os.path.exists(args.ref_xml)
    print('Generating files to: ', os.path.abspath(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    skip_geom = []
    vars_to_change = {'link_length': True,
                      'damping': True,
                      'friction': True,
                      'body_mass': True,
                      'armature': True,
                      }

    start = time.time()
    gen_robot_configs(ref_file=args.ref_xml,
                      robot_num=args.robot_num,
                      vars_to_change=vars_to_change,
                      param_var_num=args.param_var_num,
                      skip_geom=skip_geom,
                      root_save_dir=args.save_dir)
    end = time.time()
    print("Generating %d robots took %.3f seconds"
          "" % (args.robot_num, end - start))

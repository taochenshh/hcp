import argparse
import json
import os
from datetime import datetime

import numpy as np
from mujoco_py import load_model_from_path, MjSim
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--robot_file',
                        default='../../xml/simrobot/7dof/robot_mocap.xml',
                        type=str,
                        help='path to robot config')
    parser.add_argument('--save_dir', default='../../xml/gen_xmls/simrobot/reacher',
                        type=str,
                        help='path to save initial joint poses')
    print('Program starts at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))
    args = parser.parse_args()
    np.random.seed(args.seed)
    filename = 'poses.json'
    robot_file = args.robot_file
    save_dir = args.save_dir

    ini_bds = np.array([[0.3, 0.6],
                        [-0.2, 0.2],
                        [0.5, 0.7]])
    tgt_bds = np.array([[0.3, 0.6],
                        [-0.3, 0.3],
                        [-0.1, 0.3]])

    initial_num = [4, 5, 3]
    target_num = [4, 7, 5]
    target_state = np.mgrid[
                   tgt_bds[0][0]: tgt_bds[0][1]: complex(target_num[0]),
                   tgt_bds[1][0]: tgt_bds[1][1]: complex(target_num[1]),
                   tgt_bds[2][0]: tgt_bds[2][1]: complex(target_num[2])]
    target_state = target_state.reshape(3, -1).T
    initial_state = np.mgrid[
                    ini_bds[0][0]: ini_bds[0][1]: complex(initial_num[0]),
                    ini_bds[1][0]: ini_bds[1][1]: complex(initial_num[1]),
                    ini_bds[2][0]: ini_bds[2][1]: complex(initial_num[2])]
    initial_state = initial_state.reshape(3, -1).T

    np.random.shuffle(target_state)
    np.random.shuffle(initial_state)

    assert os.path.exists(robot_file)
    model = load_model_from_path(robot_file)
    sim = MjSim(model, nsubsteps=40)
    sim.reset()
    sim.step()
    site_xpos_cur = sim.data.site_xpos[0]
    print('site xpos:', site_xpos_cur)
    init_joint_angles = []
    goal_poses = []
    for idx in tqdm(range(initial_state.shape[0])):
        sim.reset()
        sim.step()
        site_xpos_target = initial_state[idx]
        delta = site_xpos_target - site_xpos_cur
        sim.data.mocap_pos[0] = sim.data.mocap_pos[0] + delta
        for i in range(30):
            sim.step()
            dist = np.linalg.norm(sim.data.site_xpos[0] - site_xpos_target)
            if dist < 0.002:
                joint_angle = sim.data.qpos[:7]
                init_joint_angles.append(joint_angle.tolist())
                break

    for idx in tqdm(range(target_state.shape[0])):
        sim.reset()
        sim.step()
        site_xpos_target = target_state[idx]
        delta = site_xpos_target - site_xpos_cur
        sim.data.mocap_pos[0] = sim.data.mocap_pos[0] + delta
        for i in range(30):
            sim.step()
            dist = np.linalg.norm(sim.data.site_xpos[0] - site_xpos_target)
            if dist < 0.002:
                goal_poses.append(site_xpos_target.tolist())
                break
    print('valid initial pose num: ', len(init_joint_angles))
    print('valid goal pose num: ', len(goal_poses))
    data = {'initial_joint_angles': init_joint_angles, 'ee_goal': goal_poses}
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, filename)
    print('saving to: ', save_file)
    with open(save_file, 'w') as f:
        json.dump(data, f, indent=2)

    print('Program ends at: \033[92m %s \033[0m' %
          datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == '__main__':
    main()

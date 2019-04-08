import argparse
import json
import os
import random
import shutil

import numpy as np
import torch
from hopper import HopperEnv
from policy import MLPPolicy
from ppo import PPO

from util import logger
from util.subproc_vec_env import SubprocVecEnv


def make_simrobot(env_name,
                  robot_dir,
                  train=True,
                  with_embed=None,
                  with_kin=None,
                  with_dyn=None,
                  train_ratio=0.8
                  ):
    train_files, test_files = train_test_split(robot_dir,
                                               train_ratio=train_ratio)
    if with_embed:
        train_files.extend(test_files)
        robot_files = train_files
        robot_files.sort()
    elif train:
        robot_files = train_files
    else:
        robot_files = test_files

    if env_name == 'hopper':
        env_fn = HopperEnv
    else:
        raise ValueError('Unknown env:', env_name)
    env = env_fn(robot_folders=robot_files,
                 robot_dir=robot_dir,
                 substeps=4,
                 train=train,
                 with_embed=with_embed,
                 with_kin=with_kin,
                 with_dyn=with_dyn)
    return env


def train_test_split(robot_dir, train_ratio=0.8):
    split_file = os.path.join(robot_dir, 'train_test_split.json')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            data = json.load(f)
        train_files = data['train']
        test_files = data['test']
    else:
        robot_files = os.listdir(robot_dir)
        robot_files = [i for i in robot_files
                       if 'robot' in i and
                       os.path.isdir(os.path.join(robot_dir, i))]
        train_num = int(len(robot_files) * train_ratio)
        random.shuffle(robot_files)
        train_files = robot_files[:train_num]
        test_files = robot_files[train_num:]
        data = {'train': train_files, 'test': test_files}
        with open(split_file, 'w') as f:
            json.dump(data, f, indent=4)
    return train_files, test_files


def make_simrobot_subproc(env_name, robot_dir,
                          train, num_env,
                          with_embed=None,
                          with_kin=None,
                          with_dyn=None,
                          train_ratio=0.8):
    def make_env(env_name, robot_dir, train,
                 with_embed=None,
                 with_kin=None,
                 with_dyn=None,
                 train_ratio=0.8):
        def _thunk():
            env = make_simrobot(env_name, robot_dir, train,
                                with_embed=with_embed,
                                with_kin=with_kin,
                                with_dyn=with_dyn,
                                train_ratio=train_ratio)
            return env

        return _thunk

    return SubprocVecEnv([make_env(env_name, robot_dir, train,
                                   with_embed=with_embed,
                                   with_kin=with_kin,
                                   with_dyn=with_dyn,
                                   train_ratio=train_ratio)
                          for i in range(num_env)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--env', help='env', default='hopper',
                        type=str, choices=['hopper'])
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_iters', type=int, default=500000)
    parser.add_argument('--save_interval', type=int, default=200,
                        help='save model every n iterations')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='logging every n iterations')
    parser.add_argument('--noptepochs', type=int, default=5,
                        help='network training epochs in each iteration')
    parser.add_argument('--nsteps', type=int, default=2048,
                        help='number of steps in an episode')
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--nminibatches', type=int, default=32,
                        help='number of minibatches in network training')
    parser.add_argument('--hid1_dim', type=int, default=128,
                        help='first hidden layer size')
    parser.add_argument('--hid2_dim', type=int, default=128,
                        help='second hidden layer size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00,
                        help='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='gamma to calculate return')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='lambda to calculate gae')
    parser.add_argument('--cliprange', type=float, default=0.2,
                        help='clip range')
    parser.add_argument('--ent_coef', type=float, default=0.015,
                        help='entropy loss coefficient')
    parser.add_argument('--ob_rms', type=bool, default=False,
                        help='calculate running mean of obs')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='value function loss coefficient')
    parser.add_argument('--with_kin', '-wk', action='store_true',
                        help='add kinematics to state')
    parser.add_argument('--with_dyn', '-wd', action='store_true',
                        help='add dynamics to state')
    parser.add_argument('--with_embed', '-we', action='store_true',
                        help='add embedding to state')
    parser.add_argument('--robot_num', type=int, default=1000,
                        help='number of robots')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm in back-propagation')
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--robot_dir', type=str,
                        default='./robots')
    parser.add_argument('--pretrain_dir', type=str,
                        default=None)
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='ratio of training robots')
    parser.add_argument('--resume', '-rt', action='store_true',
                        help='resume training')
    parser.add_argument('--resume_step', '-rs', type=int, default=None)
    parser.add_argument('--render', action='store_true',
                        help='render at testing time')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    log_dir = os.path.join(args.save_dir, 'logs')
    if not args.test and not args.resume:
        # dele = input("Do you wanna recreate ckpt and log folders? (y/n)")
        # if dele == 'y':
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'csv'])

    policy = MLPPolicy
    if args.test:
        args.num_envs = 1
    env = make_simrobot_subproc(env_name=args.env,
                                robot_dir=args.robot_dir,
                                train=not args.test,
                                num_env=args.num_envs,
                                with_embed=args.with_embed,
                                with_kin=args.with_kin,
                                with_dyn=args.with_dyn,
                                train_ratio=args.train_ratio)

    ppo = PPO(policy=policy,
              env=env,
              nsteps=args.nsteps,
              num_iters=args.num_iters,
              ent_coef=args.ent_coef,
              lr=args.lr,
              hid1_dim=int(args.hid1_dim),
              hid2_dim=int(args.hid2_dim),
              weight_decay=args.weight_decay,
              save_dir=args.save_dir,
              vf_coef=args.vf_coef,
              max_grad_norm=args.max_grad_norm,
              gamma=args.gamma,
              lam=args.lam,
              log_interval=args.log_interval,
              nminibatches=args.nminibatches,
              noptepochs=args.noptepochs,
              cliprange=args.cliprange,
              save_interval=args.save_interval,
              resume=args.resume,
              resume_step=args.resume_step,
              test=args.test,
              ob_rms=args.ob_rms,
              pretrain_dir=args.pretrain_dir,
              embed_dim=args.embed_dim,
              robot_num=args.robot_num,
              with_embed=args.with_embed)
    if args.test:
        ppo.test(render=args.render)
    else:
        ppo.train()


if __name__ == '__main__':
    main()

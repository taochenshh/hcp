import argparse
import multiprocessing as mp
import os
import shutil
from os.path import expanduser
from subprocess import Popen, DEVNULL

import numpy as np
from tqdm import tqdm


def convert_xml(args, robot_folders, inds):
    for i in tqdm(range(inds.size)):
        folder = robot_folders[inds[i]]
        robot_folder = os.path.join(args.config_dir, folder)
        robot_file = os.path.join(robot_folder, 'robot.xml')
        dest_file = os.path.join(robot_folder, 'robot.mjb')
        c_args = ("%s %s %s" % (exe_path, robot_file, dest_file))
        popen = Popen(c_args, shell=True, stdout=DEVNULL)
        popen.wait()


if __name__ == "__main__":
    desp = 'Convert mujoco xml files to mjb'
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('--config_dir', '-cd', type=str,
                        default='../xml/gen_xmls/simrobot',
                        help='directory of robot xmls')
    args = parser.parse_args()
    home = expanduser("~")
    cwd = os.getcwd()
    src_key = os.path.join(home, '.mujoco', 'mjkey.txt')
    dest_key = os.path.join(cwd, 'mjkey.txt')
    shutil.copyfile(src_key, dest_key)
    exe_path = os.path.join(home, '.mujoco', 'mjpro150',
                            'bin', 'compile')
    robot_folders = os.listdir(args.config_dir)

    num_processes = 1
    inds = np.array_split(np.arange(0,
                                    len(robot_folders)),
                          num_processes)
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=convert_xml, args=(args,
                                                 robot_folders,
                                                 inds[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

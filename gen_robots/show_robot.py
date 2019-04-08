import argparse
import os

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer


def main():
    desp = 'Display Robot'
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('--robot_file', type=str,
                        default='../xml/simrobot/7dof/peg/robot1.xml')
    args = parser.parse_args()
    np.set_printoptions(precision=6, suppress=True)
    print('Displaying robot from:', os.path.abspath(args.robot_file))
    model = load_model_from_path(args.robot_file)
    sim = MjSim(model, nsubsteps=20)
    sim.reset()
    sim.step()
    viewer = MjViewer(sim)
    while True:
        viewer.render()


if __name__ == '__main__':
    main()

## Hardware Conditioned Policies for Multi-Robot Transfer Learning
#### In NeurIPS 2018 [[Project Website]](https://sites.google.com/view/robot-transfer-hcp/home) [[Demo Video]](https://youtu.be/8odcwNOtAwI) [[pdf]](https://arxiv.org/abs/1811.09864)

[Tao Chen](https://taochenshh.github.io), [Adithya Murali](http://adithyamurali.com), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg)

The Robotics Institute, Carnegie Mellon University

<img src='images/hcp.png' width="400"> <img src="https://thumbs.gfycat.com/SafeNeighboringHydatidtapeworm-size_restricted.gif" width="400">

This is a pytorch-based implementation for our [NeurIPS 2018 paper on hardware conditioned policies](https://arxiv.org/abs/1811.09864).  The idea is that the policy input(state) is augmented with a hardware-specific encoding vector for better multi-robot skill transfer. The encoding vector can be either explicitly constructed (HCP-E) or learned implicitly via back-propagation (HCP-I). It's compatible with most of the existing deep reinforcement learning algorithms. We demonstrate the usage of our idea with DDPG+HER and PPO. If you find this work useful in your research, please cite:

	@inproceedings{chen2018hardware,
	  title={Hardware Conditioned Policies for Multi-Robot Transfer Learning},
	  author={Chen, Tao and Murali, Adithyavairavan and Gupta, Abhinav},
	  booktitle={Advances in Neural Information Processing Systems},
	  pages={9355--9366},
	  year={2018}
	}

The code has been tested on **Ubuntu 16.04**.

### Installation
1. Install [Anaconda](https://www.anaconda.com/download/#linux)

2. Download code repo:
```bash
cd ~
git clone https://github.com/taochenshh/hcp.git
cd hcp
```

3. Create python environment
```bash
conda env create -f environment.yml
conda activate hcp
```

4. Install [MuJoCo](https://www.roboti.us/index.html) and [mujoco-py 1.50](https://github.com/openai/mujoco-py)

### HCP-E Usage
1. Generate robot xml files
```bash
cd gen_robots
chmod +x gen_multi_dof_simrobot.sh
## generate both peg_insertion and reacher environments
./gen_multi_dof_simrobot.sh peg_insertion reacher
## generate peg_insertion environments only
./gen_multi_dof_simrobot.sh peg_insertion
## generate reacher environments only
./gen_multi_dof_simrobot.sh reacher
```

2. Train the policy model
```bash
cd ../HCP-E

## HCP-E: peg_insertion
python main.py --env=peg_insertion --with_kin --train_ratio=0.9 --save_interval=200 --robot_dir=../xml/gen_xmls/simrobot/peg_insertion --save_dir=peg_data/HCP-E

## HCP-E: reacher
cd util
python gen_start_and_goal.py
cd ..
python main.py --env=reacher --with_kin --train_ratio=0.9 --save_interval=200 --robot_dir=../xml/gen_xmls/simrobot/reacher --save_dir=reacher_data/HCP-E
```

3. Test the policy model
```bash
## HCP-E: peg_insertion
python main.py --env=peg_insertion --with_kin --train_ratio=0.9 --save_interval=200 --robot_dir=../xml/gen_xmls/simrobot/peg_insertion --save_dir=peg_data/HCP-E --test

## HCP-E: reacher
python main.py --env=reacher --with_kin --train_ratio=0.9 --save_interval=200 --robot_dir=../xml/gen_xmls/simrobot/reacher --save_dir=reacher_data/HCP-E --test
```
Add `--render` in the end if you want to visually test the policy.

### HCP-I Usage
1. Generate robot xml files
```bash
cd gen_robots
python gen_hoppers.py --robot_num=1000
```

2. Train the policy model
```bash
cd ../HCP-I

python main.py --env=hopper --with_embed --robot_dir=../xml/gen_xmls/hopper --save_dir=hopper_data/HCP-I
```

3. Test the policy model
```bash
python main.py --env=hopper --with_embed --robot_dir=../xml/gen_xmls/hopper --save_dir=hopper_data/HCP-I --test
```
Add `--render` in the end if you want to visually test the policy.


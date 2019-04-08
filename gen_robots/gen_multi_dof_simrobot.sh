#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

peg_insertion=false
reacher=false
for i in "$@" ; do
    if [[ $i == "peg_insertion" ]] ; then
    	peg_insertion=true
    elif [[ $i == "reacher" ]] ; then
    	reacher=true
    fi
done

## peg insertion
if $peg_insertion; then
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/7dof/peg/robot1.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/6dof/peg/robot1.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/6dof/peg/robot2.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/6dof/peg/robot3.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/peg/robot1.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/peg/robot2.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/peg/robot3.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/peg/robot4.xml --save_dir=../xml/gen_xmls/simrobot/peg_insertion
fi

## reacher
if $reacher; then
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/7dof/reacher/robot1.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/6dof/reacher/robot1.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/6dof/reacher/robot2.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/6dof/reacher/robot3.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/reacher/robot1.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/reacher/robot2.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/reacher/robot3.xml --save_dir=../xml/gen_xmls/simrobot/reacher
    python gen_simrobots.py --robot_num=200 --ref_xml=../xml/simrobot/5dof/reacher/robot4.xml --save_dir=../xml/gen_xmls/simrobot/reacher
fi

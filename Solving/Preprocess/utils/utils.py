import numpy as np
import json
import os
import random
import h5py
import torch
import torch.nn as nn

BODY_JOINT_NAME_LIST_PATH = "configs/joint_config_body.txt"
FULL_BODY_JOINT_CONFIG_PATH = "configs/joint_config_full_body.txt"
LEFTHAND_JOINT_CONFIG_PATH = "configs/joint_config_left_hand.txt"
RIGHTHAND_JOINT_CONFIG_PATH = "configs/joint_config_right_hand.txt"
MARKER_NAME_CONFIG_PATH = "configs/marker_names.json"
with open(FULL_BODY_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    FULL_BODY_JOINT_NAME_LIST = [line.split()[0] for line in content]
    FULL_BODY_JOINT_PARENT = [int(line.split()[1]) for line in content]
    # FULL_BODY_END_JOINT_IDX = [14, 19, 21, 22, 23]

with open(BODY_JOINT_NAME_LIST_PATH, "r") as f:
    content = f.readlines()
    BODY_JOINT_NAME_LIST = [line.split()[0] for line in content]
    BODY_JOINT_PARENT = [int(line.split()[1]) for line in content]
    BODY_END_JOINT_IDX = [16, 17, 24, 25, 26]

with open(LEFTHAND_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    LEFTHAND_JOINT_NAME_LIST = [line.split()[0] for line in content]
    LEFTHAND_JOINT_PARENT = [int(line.split()[1]) for line in content]
    LEFTHAND_END_JOINT_IDX = [14, 19, 21, 22, 23]

with open(RIGHTHAND_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    RIGHTHAND_JOINT_NAME_LIST = [line.split()[0] for line in content]
    RIGHTHAND_JOINT_PARENT = [int(line.split()[1]) for line in content]
    RIGHTHAND_END_JOINT_IDX = [14, 19, 21, 22, 23]

SYNTHETIC_BODY_JOINT_NAME_LIST_PATH = "configs/synthetic_joint_config_body.txt"
SYNTHETIC_FULL_BODY_JOINT_CONFIG_PATH = "configs/synthetic_joint_config_full.txt"
SYNTHETIC_LEFTHAND_JOINT_CONFIG_PATH = "configs/synthetic_joint_config_left_hand.txt"
SYNTHETIC_RIGHTHAND_JOINT_CONFIG_PATH = "configs/synthetic_joint_config_right_hand.txt"

with open(SYNTHETIC_FULL_BODY_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    SYNTHETIC_FULL_BODY_JOINT_NAME_LIST = [line.split()[0] for line in content]
    SYNTHETIC_FULL_BODY_JOINT_PARENT = [int(line.split()[1]) for line in content]
    # FULL_BODY_END_JOINT_IDX = [14, 19, 21, 22, 23]

with open(SYNTHETIC_BODY_JOINT_NAME_LIST_PATH, "r") as f:
    content = f.readlines()
    SYNTHETIC_BODY_JOINT_NAME_LIST = [line.split()[0] for line in content]
    SYNTHETIC_BODY_JOINT_PARENT = [int(line.split()[1]) for line in content]
    SYNTHETIC_BODY_END_JOINT_IDX = [16, 17, 24, 25, 26]

with open(SYNTHETIC_LEFTHAND_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    SYNTHETIC_LEFTHAND_JOINT_NAME_LIST = [line.split()[0] for line in content]
    SYNTHETIC_LEFTHAND_JOINT_PARENT = [int(line.split()[1]) for line in content]
    SYNTHETIC_LEFTHAND_END_JOINT_IDX = [14, 19, 21, 22, 23]

with open(SYNTHETIC_RIGHTHAND_JOINT_CONFIG_PATH, "r") as f:
    content = f.readlines()
    SYNTHETIC_RIGHTHAND_JOINT_NAME_LIST = [line.split()[0] for line in content]
    SYNTHETIC_RIGHTHAND_JOINT_PARENT = [int(line.split()[1]) for line in content]
    SYNTHETIC_RIGHTHAND_END_JOINT_IDX = [14, 19, 21, 22, 23]

marker_name_config = json.load(open(MARKER_NAME_CONFIG_PATH, "r"))

import copy
import os
import random

import h5py
import numpy as np
import torch
import time
import subprocess
import argparse
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import pandas as pd

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from Network.utils.pymo.parsers import BVHParser
from Network.utils.pymo.writers import BVHWriter
from Network.utils.pymo.data import MocapData

from Network.models import *
from Network.utils.average_meter import AverageMeter
from Network.utils.marker_utils import *

MARKER_NAME_CONFIG_PATH = "configs/marker_names.json"
PRODUCTION_WAIST_MARKER_FILE = "configs/production_up_waist.npy"
FRONT_WAIST_WAIST_MARKER_FILE = "configs/front_up_waist.npy"

PRODUCTION_WAIST_MARKER_NAMES = ["LFWT", "LMWT", "LBWT", "RFWT", "RMWT", "RBWT", "STRN", "T10"]
FRONT_WAIST_MARKER_NAMES = ["LFWT", "MFWT", "RFWT", "LBWT", "MBWT", "RBWT", "STRN", "T10"]
LEFT_WRIST_MARKER_NAMES = ["LIWR", "LOWR", "LIHAND", "LOHAND"]
RIGHT_WRIST_MARKER_NAMES = ["RIWR", "ROWR", "RIHAND", "ROHAND"]

PRE_T_POSE_LENGTH = 0
WINDOW_SIZE = 64
PART_SIZE = 512

marker_name_config = json.load(open(MARKER_NAME_CONFIG_PATH, "r"))
marker_name_dict = dict()
for body_type in marker_name_config["body"].keys():
    for finger_type in marker_name_config["finger"].keys():
        marker_set = frozenset(marker_name_config["body"][body_type]).union(
            frozenset(marker_name_config["finger"][finger_type]))
        marker_name_dict[marker_set] = {"body": body_type, "finger": finger_type}

production_waist_marker_pos = np.load(PRODUCTION_WAIST_MARKER_FILE)
front_waist_marker_pos = np.load(FRONT_WAIST_WAIST_MARKER_FILE)
production_waist_marker_pos -= np.mean(production_waist_marker_pos, axis=0)
front_waist_marker_pos -= np.mean(front_waist_marker_pos, axis=0)


def construct_transform_matrix(rot_mat_arr, trans_arr):
    # rotation -> translation
    assert rot_mat_arr.shape[0] == trans_arr.shape[0]
    assert rot_mat_arr.shape[1] == 3 and rot_mat_arr.shape[2] == 3 and trans_arr.shape[1] == 3
    anim_length = rot_mat_arr.shape[0]
    eye_mat_arr = np.repeat(np.eye(4).reshape((1, 4, 4)), anim_length, axis=0)
    trans = copy.deepcopy(eye_mat_arr)
    rot = copy.deepcopy(eye_mat_arr)
    rot[:, :3, :3] = rot_mat_arr
    trans[:, :3, 3] = trans_arr

    return trans @ rot


def export_to_bvh(bvh_path, joint_names, rot_mat_arr, trans, t_pose_joint_pos, parent):
    out_data = MocapData()
    anim_length = rot_mat_arr.shape[0]

    is_end_arr = [len(list(np.asarray(joint_names)[np.where(parent == i)])) == 0 for i in
                  range(len(joint_names))]
    skeleton_joint_name_arr = [joint_names[parent[i]] + "_Nub" if is_end_arr[i] else joint_names[i]
                               for i in range(len(joint_names))]

    out_data.channel_names.append((joint_names[0], "Xposition"))
    out_data.channel_names.append((joint_names[0], "Yposition"))
    out_data.channel_names.append((joint_names[0], "Zposition"))

    for joint_idx, name in enumerate(joint_names):
        if not is_end_arr[joint_idx]:
            out_data.channel_names.append((name, "Zrotation"))
            out_data.channel_names.append((name, "Xrotation"))
            out_data.channel_names.append((name, "Yrotation"))

    out_data.framerate = 1 / 120
    out_data.root_name = joint_names[0]

    for joint_idx, name in enumerate(joint_names):
        children = list(np.asarray(skeleton_joint_name_arr)[np.where(parent == joint_idx)])
        skeleton_joint_name = skeleton_joint_name_arr[joint_idx]
        is_end = is_end_arr[joint_idx]

        out_data.skeleton[skeleton_joint_name] = {}
        if parent[joint_idx] != -1:
            out_data.skeleton[skeleton_joint_name]["parent"] = joint_names[parent[joint_idx]]
        else:
            out_data.skeleton[skeleton_joint_name]["parent"] = None

        if joint_idx == 0:
            out_data.skeleton[skeleton_joint_name]["channels"] = ["Xposition", "Yposition", "Zposition",
                                                                  "Zrotation", "Xrotation", "Yrotation"]
        elif not is_end:
            out_data.skeleton[skeleton_joint_name]["channels"] = ["Zrotation", "Xrotation", "Yrotation"]
        else:
            out_data.skeleton[skeleton_joint_name]["channels"] = []

        if joint_idx == 0:
            out_data.skeleton[skeleton_joint_name]["offsets"] = list(trans[0])
        else:
            out_data.skeleton[skeleton_joint_name]["offsets"] = \
                list(t_pose_joint_pos[joint_idx] - t_pose_joint_pos[parent[joint_idx]])

        out_data.skeleton[skeleton_joint_name]["children"] = children

    time_index = pd.to_timedelta([i * out_data.framerate for i in range(anim_length)], unit="s")
    column_names = ["{}_{}".format(name[0], name[1]) for name in out_data.channel_names]
    channels = np.zeros((anim_length, len(column_names)))

    channels[:, :3] = trans
    channel_idx = 1
    for i in range(len(joint_names)):
        if not is_end_arr[i]:
            channels[:, 3 * channel_idx:  3 * (channel_idx + 1)] = \
                Rotation.from_matrix(rot_mat_arr[:, i]).as_euler("ZXY", degrees=True)
            channel_idx += 1

    out_data.values = pd.DataFrame(data=channels, index=time_index, columns=column_names)

    writer = BVHWriter()
    writer.write(out_data, open(bvh_path, "w"))


def get_joint_transform_matrix(rot_mat, t_pose_joint_pos, joint_hierarchy):
    def with_zeros(x):
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)))

    def pack(x):
        return np.dstack((np.zeros((x.shape[0], 4, 3), dtype=np.float64), x))

    joint_num = rot_mat.shape[0]
    R = rot_mat

    G = np.empty((joint_num, 4, 4))
    G[0] = with_zeros(np.hstack((R[0], t_pose_joint_pos[0].reshape((3, 1)))))
    for i in range(1, joint_num):
        G[i] = np.matmul(G[joint_hierarchy[i]],
                         with_zeros(np.hstack((R[i], (t_pose_joint_pos[i] - t_pose_joint_pos[joint_hierarchy[i]]). \
                                               reshape((3, 1))))))
    global_joint_pos = G[:, :3, 3]
    # G = G - pack(np.matmul(G, np.hstack((t_pose_joint_pos, np.zeros((joint_num, 1)))).reshape((joint_num, 4, 1))))
    return G, global_joint_pos


def calc_align_rot_matrix(vec1, vec2):
    axis = np.cross(vec1, vec2)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    mat = Rotation.from_mrp(np.tan(angle / 4) * axis).as_matrix()
    return mat


def align_body_markers(body_marker_pos_arr, waist_marker_pos_arr, ref_waist_marker_pos, character_type):
    def get_waist_axis(waist_marker_pos, character_type):
        up = np.mean(waist_marker_pos[6:], axis=0) - np.mean(waist_marker_pos[:6], axis=0)
        up /= np.linalg.norm(up)
        if character_type == "production":
            forward = ((waist_marker_pos[6] - waist_marker_pos[7]) + \
                       (waist_marker_pos[0] - waist_marker_pos[2]) + \
                       (waist_marker_pos[3] - waist_marker_pos[5])) / 3

        elif character_type == "front_waist":
            forward = ((waist_marker_pos[6] - waist_marker_pos[7]) + \
                       (waist_marker_pos[2] - waist_marker_pos[5]) + \
                       (waist_marker_pos[1] - waist_marker_pos[4]) + \
                       (waist_marker_pos[0] - waist_marker_pos[3])) / 4
        forward -= up * np.dot(forward, up)
        forward /= np.linalg.norm(forward)
        left = np.cross(up, forward)
        return np.vstack([up, forward, left]).T

    def get_global_transformation_arr(waist_marker_pos_arr, ref_waist_marker_pos, character_type, align_type):
        anim_length = waist_marker_pos_arr.shape[0]
        ref_waist_axis = get_waist_axis(ref_waist_marker_pos, character_type)
        global_rotation_arr = []

        global_translation_arr = np.mean(waist_marker_pos_arr, axis=1) - np.mean(ref_waist_marker_pos, axis=0)

        for frame in range(anim_length):
            waist_marker_axis = get_waist_axis(waist_marker_pos_arr[frame] - global_translation_arr[frame],
                                               character_type)
            if align_type == "up":
                R = calc_align_rot_matrix(np.asarray([0, 1, 0]), waist_marker_axis[:, 0])
            elif align_type == "horizontal":
                R = calc_align_rot_matrix(np.asarray([0, 0, 1]), waist_marker_axis[:, 1])
            global_rotation_arr.append(R)

        global_rotation_arr = np.asarray(global_rotation_arr)
        return global_rotation_arr, global_translation_arr

    up_align_rotation_arr, global_translation_arr = \
        get_global_transformation_arr(waist_marker_pos_arr,
                                      ref_waist_marker_pos,
                                      character_type,
                                      align_type="up")
    up_aligned_waist_markers = \
        remove_marker_global_transform(waist_marker_pos_arr,
                                       up_align_rotation_arr,
                                       global_translation_arr)
    horizontal_rot_align_rotation_arr, _ = \
        get_global_transformation_arr(up_aligned_waist_markers,
                                      ref_waist_marker_pos,
                                      character_type, align_type="horizontal")
    body_marker_without_global_transform = \
        remove_marker_global_transform(body_marker_pos_arr,
                                       up_align_rotation_arr @ horizontal_rot_align_rotation_arr,
                                       global_translation_arr)
    return body_marker_without_global_transform, \
        up_align_rotation_arr @ horizontal_rot_align_rotation_arr, global_translation_arr


def align_hand_markers(finger_marker_pos_arr, wrist_marker_pos_arr):
    def get_wrist_axis(wrist_marker_pos):
        up = np.mean(wrist_marker_pos[2:], axis=0) - np.mean(wrist_marker_pos[:2], axis=0)
        up = up / np.linalg.norm(up)
        out = (wrist_marker_pos[1] + wrist_marker_pos[3] - wrist_marker_pos[0] - wrist_marker_pos[2]) / 2
        out -= up * np.dot(out, up)
        out = out / np.linalg.norm(out)
        forward = np.cross(up, out)
        res = np.vstack([up, forward, out]).T
        return res

    def get_hand_global_transformation_arr(finger_marker_pos_arr, wrist_marker_pos_arr, align_type):
        anim_length = wrist_marker_pos_arr.shape[0]
        global_translation_arr = np.mean(wrist_marker_pos_arr, axis=1)
        global_rotation_arr = []
        for frame in range(anim_length):
            wrist_marker_axis = get_wrist_axis(wrist_marker_pos_arr[frame] - global_translation_arr[frame])
            if align_type == "up":
                R = calc_align_rot_matrix(np.asarray([0, 1, 0]), wrist_marker_axis[:, 0])
            if align_type == "horizontal":
                R = calc_align_rot_matrix(np.asarray([0, 0, 1]), wrist_marker_axis[:, 1])
            global_rotation_arr.append(R)
        global_rotation_arr = np.asarray(global_rotation_arr)

        return global_rotation_arr, global_translation_arr

    up_rotation_arr, translation_arr = get_hand_global_transformation_arr(
        finger_marker_pos_arr,
        wrist_marker_pos_arr, "up")
    up_aligned_finger_marker_pos = \
        remove_marker_global_transform(finger_marker_pos_arr,
                                       up_rotation_arr,
                                       translation_arr)
    up_aligned_wrist_marker_pos = \
        remove_marker_global_transform(wrist_marker_pos_arr,
                                       up_rotation_arr,
                                       translation_arr)
    horizontal_rotation_arr, _ = \
        get_hand_global_transformation_arr(
            up_aligned_finger_marker_pos,
            up_aligned_wrist_marker_pos, "horizontal")

    finger_marker_pos_without_global_transform = \
        remove_marker_global_transform(finger_marker_pos_arr,
                                       up_rotation_arr @ horizontal_rotation_arr,
                                       translation_arr)

    return finger_marker_pos_without_global_transform, \
        up_rotation_arr @ horizontal_rotation_arr, translation_arr


def get_character_marker_pos_vis_index(all_marker_names, marker_pos, marker_vis,
                                       marker_name_dict, marker_name_config):
    valid_marker_names = list(filter(lambda x: (":" in x and x[0] != "*" and x.find("LCM") == -1), all_marker_names))
    character_list = list(set(list(map(lambda x: x[:x.find(":")], valid_marker_names))))
    valid_character_list = []
    for character in character_list:
        corresponding_markers = list(filter(lambda x: x[:x.find(":")] == character and \
                                                      x.find("LCM") == -1, all_marker_names))
        marker_name = list(map(lambda x: x[len(character) + 1:], corresponding_markers))
        if frozenset(marker_name) in marker_name_dict:
            valid_character_list.append(character)
    if len(valid_character_list) == 0:
        raise ValueError("No character has a valid marker config")

    character_marker_config_list = []
    character_marker_pos_list = []
    character_marker_vis_list = []
    character_marker_index_list = []
    character_marker_name_list = []
    for character in valid_character_list:
        corresponding_markers = list(filter(lambda x: x[:x.find(":")] == character and \
                                                      x.find("LCM") == -1, all_marker_names))
        marker_name = list(map(lambda x: x[len(character) + 1:], corresponding_markers))
        marker_config = marker_name_dict[frozenset(marker_name)]

        character_marker_config_list.append(marker_config)
        marker_list = marker_name_config["body"][marker_config["body"]] + \
                      marker_name_config["finger"][marker_config["finger"]]
        marker_names = ["{}:{}".format(character, marker_name) for marker_name in marker_list]
        marker_index = [all_marker_names.index(marker_name) for marker_name in marker_names]
        character_marker_pos = marker_pos[:, marker_index, :]
        character_marker_vis = marker_vis[:, marker_index]

        character_marker_index_list.append(marker_index)
        character_marker_pos_list.append(character_marker_pos)
        character_marker_vis_list.append(character_marker_vis)
        character_marker_name_list.append(marker_names)
    return valid_character_list, character_marker_config_list, \
        character_marker_pos_list, character_marker_vis_list, \
        character_marker_index_list, character_marker_name_list


def batch_generate_window_data(arrs, window_size):
    ret = []
    for arr in arrs:
        assert arr.shape[0] == arrs[0].shape[0]
    anim_length = arrs[0].shape[0]
    step_size = int(window_size / 2.0)
    window_num = int(np.ceil(anim_length / float(step_size)))

    for arr in arrs:
        window_arr = np.zeros((window_num, window_size, *arr.shape[1:]))
        for window_idx in range(window_num):
            for window_frame in range(window_size):
                ii = min(anim_length - 1, window_idx * step_size + window_frame)
                window_arr[window_idx, window_frame] = arr[ii]
        ret.append(window_arr)
    if len(arrs) == 1:
        return ret[0]
    else:
        return ret


def cat_window_arr(arr):
    window_num, window_size = arr.shape[0], arr.shape[1]
    step_size = window_size // 2
    anim_length = (window_num + 1) * step_size
    if type(arr) == torch.Tensor:
        out = torch.zeros((anim_length, *arr.shape[2:])).to(arr.device)
    else:
        out = np.zeros((anim_length, *arr.shape[2:]))
    for frame in range(anim_length):
        window_idx = frame // step_size
        step_idx = frame % step_size
        if window_idx == 0:
            out[frame] = arr[0, frame]
        elif window_idx == window_num:
            out[frame] = arr[window_idx - 1, step_idx + step_size]
        else:
            out[frame] = 0.5 * (arr[window_idx, step_idx] + \
                                arr[window_idx - 1, step_idx + step_size])
    return out


def marker_to_pose(args, intermediate_file, out_file, verbose=True):
    npz_content = dict(np.load(intermediate_file))

    marker_names = marker_name_config["body"]["front_waist"] + marker_name_config["finger"]["10"]
    marker_vis = npz_content["corrupted_marker_vis"]
    marker_pos = npz_content["network_fixed_marker_pos"]

    marker_names = ["character:{}".format(name) for name in marker_names]

    character_list, character_marker_config_list, character_marker_pos_list, \
        character_marker_vis_list, character_marker_index_list, character_marker_name_list = \
        get_character_marker_pos_vis_index(marker_names, marker_pos, marker_vis,
                                           marker_name_dict, marker_name_config)

    for character_idx in range(len(character_list)):
        character_name = character_list[character_idx]
        character_config = character_marker_config_list[character_idx]
        character_marker_pos = character_marker_pos_list[character_idx]
        character_marker_vis = character_marker_vis_list[character_idx] * -1
        character_marker_name = character_marker_name_list[character_idx]
        character_marker_index = character_marker_index_list[character_idx]
        start_time = time.time()

        print("Processing character {}, body marker config {}, have {} hand markers". \
              format(character_name, character_config["body"], character_config["finger"]))

        character_marker_pos = character_marker_pos[:]
        character_marker_vis = character_marker_vis[:]

        anim_length = character_marker_pos.shape[0]

        body_marker_names = marker_name_config["body"][character_config["body"]]
        hand_marker_names = marker_name_config["finger"][character_config["finger"]]
        marker_names = body_marker_names + hand_marker_names
        marker_num = len(marker_names)

        left_hand_marker_names = list(filter(lambda name: name[0] == "L",
                                             hand_marker_names))
        right_hand_marker_names = list(filter(lambda name: name[0] == "R",
                                              hand_marker_names))
        body_marker_num = len(body_marker_names)
        hand_marker_num = int(len(hand_marker_names) / 2)
        have_hand = (hand_marker_num != 0)

        body_marker_index = [character_marker_name.index("{}:{}".format(character_name, marker_name)) \
                             for marker_name in body_marker_names]
        left_hand_marker_index = [character_marker_name.index("{}:{}".format(character_name, marker_name)) \
                                  for marker_name in left_hand_marker_names]
        right_hand_marker_index = [character_marker_name.index("{}:{}".format(character_name, marker_name)) \
                                   for marker_name in right_hand_marker_names]
        left_wrist_marker_index = [character_marker_name.index("{}:{}".format(character_name, marker_name)) \
                                   for marker_name in LEFT_WRIST_MARKER_NAMES]
        right_wrist_marker_index = [character_marker_name.index("{}:{}".format(character_name, marker_name)) \
                                    for marker_name in RIGHT_WRIST_MARKER_NAMES]

        body_marker_pos = character_marker_pos[:, body_marker_index]
        left_hand_marker_pos = character_marker_pos[:, left_hand_marker_index]
        right_hand_marker_pos = character_marker_pos[:, right_hand_marker_index]
        left_wrist_marker_pos = character_marker_pos[:, left_wrist_marker_index]
        right_wrist_marker_pos = character_marker_pos[:, right_wrist_marker_index]

        body_marker_vis = character_marker_vis[:, body_marker_index].astype(int) * -1
        left_hand_marker_vis = character_marker_vis[:, left_hand_marker_index].astype(int) * -1
        right_hand_marker_vis = character_marker_vis[:, right_hand_marker_index].astype(int) * -1

        if character_config["body"] == "front_waist":
            waist_marker_index = [character_marker_name.index("{}:{}".format(character_name, marker_name)) \
                                  for marker_name in FRONT_WAIST_MARKER_NAMES]
            waist_marker_pos_arr = character_marker_pos[:, waist_marker_index]
            ref_waist_marker_pos = front_waist_marker_pos

        _, global_body_rot_mat, global_body_trans = align_body_markers(
            body_marker_pos, waist_marker_pos_arr,
            ref_waist_marker_pos, character_config["body"])

        if have_hand:
            _, global_left_hand_rot_mat, global_left_hand_trans = \
                align_hand_markers(left_hand_marker_pos,
                                   left_wrist_marker_pos)
            _, global_right_hand_rot_mat, global_right_hand_trans = \
                align_hand_markers(right_hand_marker_pos,
                                   right_wrist_marker_pos)

        aligned_body_marker_pos = remove_marker_global_transform(body_marker_pos,
                                                                 global_body_rot_mat,
                                                                 global_body_trans)
        if have_hand:
            aligned_left_hand_marker_pos = remove_marker_global_transform(left_hand_marker_pos,
                                                                          global_left_hand_rot_mat,
                                                                          global_left_hand_trans)
            aligned_right_hand_marker_pos = remove_marker_global_transform(right_hand_marker_pos,
                                                                           global_right_hand_rot_mat,
                                                                           global_right_hand_trans)

            aligned_left_wrist_marker_pos = remove_marker_global_transform(left_wrist_marker_pos,
                                                                           global_left_hand_rot_mat,
                                                                           global_left_hand_trans)
            aligned_right_wrist_marker_pos = remove_marker_global_transform(right_wrist_marker_pos,
                                                                            global_right_hand_rot_mat,
                                                                            global_right_hand_trans)

        window_body_marker_pos, \
            window_left_hand_marker_pos, \
            window_right_hand_marker_pos, \
            window_left_wrist_marker_pos, \
            window_right_wrist_marker_pos, \
            window_body_marker_vis, \
            window_left_hand_marker_vis, \
            window_right_hand_marker_vis, \
            window_global_body_rot_mat, \
            window_global_body_trans = \
            batch_generate_window_data([aligned_body_marker_pos,
                                        aligned_left_hand_marker_pos,
                                        aligned_right_hand_marker_pos,
                                        aligned_left_wrist_marker_pos,
                                        aligned_right_wrist_marker_pos,
                                        body_marker_vis,
                                        left_hand_marker_vis,
                                        right_hand_marker_vis,
                                        global_body_rot_mat,
                                        global_body_trans],
                                       WINDOW_SIZE)

        ## Networks
        with open(os.path.join(args.root_dir, args.body_joint_name_config_path), "r") as f:
            content = f.readlines()
            joint_list = [line.split()[0] for line in content]
            parent_list = [int(line.split()[1]) for line in content]
        body_joint_num = len(joint_list)
        body_joint_name_list = joint_list
        body_parent_list = parent_list

        with open(os.path.join(args.root_dir,
                               args.left_hand_joint_name_config_path), "r") as f:
            content = f.readlines()
            joint_list = [line.split()[0] for line in content]
            parent_list = [int(line.split()[1]) for line in content]
            joint_name_list = joint_list
        left_hand_joint_num = len(joint_list)
        left_hand_joint_name_list = joint_name_list
        left_hand_parent_list = parent_list

        with open(os.path.join(args.root_dir,
                               args.right_hand_joint_name_config_path), "r") as f:
            content = f.readlines()
            joint_list = [line.split()[0] for line in content]
            parent_list = [int(line.split()[1]) for line in content]
            joint_name_list = joint_list
        right_hand_joint_num = len(joint_list)
        right_hand_joint_name_list = joint_name_list
        right_hand_parent_list = parent_list

        if args.cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        if character_config["body"] == "front_waist":
            body_marker_pos_mean_std_path = args.front_waist_body_marker_pos_mean_std_path
            body_motion_mean_std_path = args.front_waist_body_motion_mean_std_path
            body_joint_offset_mean_std_path = args.front_waist_body_joint_offset_mean_std_path
            body_network_resume_file_path = args.front_waist_body_resume_file_path

        args.network_dim[0] = body_marker_num * args.marker_gcn_dim[-1]
        args.network_dim[-1] = (body_joint_num + 1) * args.skeleton_gcn_dim[0]
        args.skeleton_gcn_dim[-1] = 9 + 3
        args.marker_num = body_marker_num
        args.joint_num = body_joint_num
        args.parent_list = body_parent_list
        args.joint_name_list = body_joint_name_list

        body_network = GraphResNetModel(args, args.body_connect_mat_path, args.front_waist_body_weight_mean_std_path)

        body_checkpoint = torch.load(body_network_resume_file_path)
        body_network.load_state_dict(body_checkpoint)
        body_network = body_network.to(device)
        body_network.eval()

        body_motion_mean_std = np.load(body_motion_mean_std_path)
        body_motion_mean = body_motion_mean_std["mean"].astype(np.float32)
        body_motion_std = body_motion_mean_std["std"].astype(np.float32)
        body_motion_mean = body_motion_mean.reshape((1, 1, body_motion_mean.shape[0])).astype(np.float32)
        body_motion_std = body_motion_std.reshape(body_motion_mean.shape).astype(np.float32)

        body_marker_pos_mean_std = np.load(body_marker_pos_mean_std_path)
        body_marker_pos_mean = body_marker_pos_mean_std["mean"]
        body_marker_pos_std = body_marker_pos_mean_std["std"]
        body_marker_pos_mean = body_marker_pos_mean. \
            reshape((1, 1, body_marker_pos_mean.shape[0], body_marker_pos_mean.shape[1])).astype(np.float32)
        body_marker_pos_std = body_marker_pos_std.reshape(body_marker_pos_mean.shape).astype(np.float32)

        body_joint_offset_mean_std = np.load(body_joint_offset_mean_std_path)
        body_joint_offset_mean = body_joint_offset_mean_std["mean"].astype(np.float32)
        body_joint_offset_std = body_joint_offset_mean_std["std"].astype(np.float32)
        body_joint_offset_mean = body_joint_offset_mean.reshape(
            (1, body_joint_offset_mean.shape[0], body_joint_offset_mean.shape[1]))
        body_joint_offset_std = body_joint_offset_std.reshape(body_joint_offset_mean.shape)

        body_marker_pos_mean = torch.from_numpy(body_marker_pos_mean).to(device)
        body_marker_pos_std = torch.from_numpy(body_marker_pos_std).to(device)
        body_motion_mean = torch.from_numpy(body_motion_mean).to(device)
        body_motion_std = torch.from_numpy(body_motion_std).to(device)
        body_joint_offset_mean = torch.from_numpy(body_joint_offset_mean).to(device)
        body_joint_offset_std = torch.from_numpy(body_joint_offset_std).to(device)

        window_body_marker_pos = torch.from_numpy(window_body_marker_pos.astype(np.float32)).to(device)

        batch_size, window_shape = \
            window_body_marker_pos.shape[0], \
                window_body_marker_pos.shape[1]

        normed_body_marker_pos = \
            (window_body_marker_pos - body_marker_pos_mean) / body_marker_pos_std
        window_body_marker_vis_tensor = torch.from_numpy(window_body_marker_vis).to(device)

        with torch.no_grad():
            network_input = torch.cat([normed_body_marker_pos,
                                       torch.unsqueeze(window_body_marker_vis_tensor, -1)], dim=-1).float()
            pred = body_network(network_input)
            pred = pred.view((batch_size, window_shape, -1))

        pred_body_motion = torch.cat([pred[:, :, :-3].view((batch_size, window_shape, body_joint_num, -1))[:, :, :, :9].
                                     reshape((batch_size, window_shape, -1)), pred[:, :, -3:]], dim=-1)
        pred_body_joint_offset = pred[:, :, :-3].view((batch_size, window_shape, body_joint_num, -1))[:, :, :, -3:]. \
            reshape((batch_size, window_shape, -1, 3))

        pred_mean_body_joint_offset = torch.mean(pred_body_joint_offset,
                                                 dim=[0, 1]) * body_joint_offset_std + body_joint_offset_mean
        pred_mean_body_joint_offset = pred_mean_body_joint_offset[0].cpu().numpy()
        pred_mean_body_joint_offset[0] *= 0

        pred_body_motion = pred_body_motion * body_motion_std + body_motion_mean
        cated_pred_body_motion = cat_window_arr(pred_body_motion)
        cated_pred_body_rot_mat = cated_pred_body_motion[:, :-3].reshape((cated_pred_body_motion.shape[0], -1, 3, 3))
        cated_pred_body_trans = cated_pred_body_motion[:, -3:].cpu().numpy()

        ortho_pred_rot = cated_pred_body_rot_mat.clone().detach()

        ortho_pred_rot[..., 1] -= \
            (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 1]) /
             torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 0])). \
                view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 0]
        ortho_pred_rot[..., 2] -= \
            (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 2]) /
             torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 0])). \
                view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 0] + \
            (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 1], ortho_pred_rot[..., 2]) /
             torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 1], ortho_pred_rot[..., 1])). \
                view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 1]
        ortho_pred_rot[..., 0] /= torch.linalg.norm(ortho_pred_rot[..., 0], dim=-1, keepdim=True)
        ortho_pred_rot[..., 1] /= torch.linalg.norm(ortho_pred_rot[..., 1], dim=-1, keepdim=True)
        ortho_pred_rot[..., 2] /= torch.linalg.norm(ortho_pred_rot[..., 2], dim=-1, keepdim=True)

        ortho_pred_rot = ortho_pred_rot.cpu().numpy()
        pred_body_t_pose_joint_pos = np.zeros_like(pred_mean_body_joint_offset)
        for i in range(1, len(body_parent_list)):
            pred_body_t_pose_joint_pos[i] = pred_body_t_pose_joint_pos[body_parent_list[i]] + \
                                            pred_mean_body_joint_offset[i]

        align_body_transform_mat_arr = construct_transform_matrix(cat_window_arr(window_global_body_rot_mat),
                                                                  cat_window_arr(window_global_body_trans))
        pred_aligned_body_transform_mat_arr = construct_transform_matrix(ortho_pred_rot[:, 0],
                                                                         cated_pred_body_trans)
        pred_body_transform_mat_arr = align_body_transform_mat_arr @ pred_aligned_body_transform_mat_arr

        ortho_pred_rot[:, 0] = pred_body_transform_mat_arr[:, :3, :3]
        pred_body_trans = pred_body_transform_mat_arr[:, :3, 3]

        body_rot_mat = ortho_pred_rot
        body_joint_offset = pred_mean_body_joint_offset

        # np.save("out.npy",body_marker_pos)
        # export_to_bvh(out_file, body_joint_name_list, ortho_pred_rot,
        #               pred_body_trans, pred_body_t_pose_joint_pos, np.asarray(body_parent_list))

        if have_hand:
            if hand_marker_num == 10:
                left_hand_marker_pos_mean_std_path = args.ten_finger_left_hand_marker_pos_mean_std_path
                left_wrist_marker_pos_mean_std_path = args.ten_finger_left_wrist_marker_pos_mean_std_path
                left_hand_motion_mean_std_path = args.ten_finger_left_hand_motion_mean_std_path
                left_hand_joint_offset_mean_std_path = args.ten_finger_left_hand_joint_offset_mean_std_path
                left_hand_network_resume_file_path = args.ten_finger_left_hand_resume_file_path

                right_hand_marker_pos_mean_std_path = args.ten_finger_right_hand_marker_pos_mean_std_path
                right_wrist_marker_pos_mean_std_path = args.ten_finger_right_wrist_marker_pos_mean_std_path
                right_hand_motion_mean_std_path = args.ten_finger_right_hand_motion_mean_std_path
                right_hand_joint_offset_mean_std_path = args.ten_finger_right_hand_joint_offset_mean_std_path
                right_hand_network_resume_file_path = args.ten_finger_right_hand_resume_file_path

            args.network_dim[0] = (hand_marker_num + 4) * args.marker_gcn_dim[-1]
            args.network_dim[-1] = (left_hand_joint_num + 1) * args.skeleton_gcn_dim[0]
            args.skeleton_gcn_dim[-1] = 9 + 3
            args.marker_num = hand_marker_num + 4
            args.joint_num = left_hand_joint_num
            args.parent_list = left_hand_parent_list
            args.joint_name_list = left_hand_parent_list

            left_hand_network = GraphResNetModel(args, args.hand_connect_mat_path,
                                                 args.ten_finger_left_hand_weight_mean_std_path)

            left_hand_checkpoint = torch.load(left_hand_network_resume_file_path)
            left_hand_network.load_state_dict(left_hand_checkpoint)
            left_hand_network = left_hand_network.to(device)
            left_hand_network.eval()

            left_hand_motion_mean_std = np.load(left_hand_motion_mean_std_path)
            left_hand_motion_mean = left_hand_motion_mean_std["mean"].astype(np.float32)
            left_hand_motion_std = left_hand_motion_mean_std["std"].astype(np.float32)
            left_hand_motion_mean = left_hand_motion_mean.reshape((1, 1, left_hand_motion_mean.shape[0])).astype(
                np.float32)
            left_hand_motion_std = left_hand_motion_std.reshape(left_hand_motion_mean.shape).astype(np.float32)

            left_hand_marker_pos_mean_std = np.load(left_hand_marker_pos_mean_std_path)
            left_hand_marker_pos_mean = left_hand_marker_pos_mean_std["mean"]
            left_hand_marker_pos_std = left_hand_marker_pos_mean_std["std"]
            left_hand_marker_pos_mean = left_hand_marker_pos_mean. \
                reshape((1, 1, left_hand_marker_pos_mean.shape[0], left_hand_marker_pos_mean.shape[1])).astype(
                np.float32)
            left_hand_marker_pos_std = left_hand_marker_pos_std.reshape(left_hand_marker_pos_mean.shape).astype(
                np.float32)

            left_hand_joint_offset_mean_std = np.load(left_hand_joint_offset_mean_std_path)
            left_hand_joint_offset_mean = left_hand_joint_offset_mean_std["mean"].astype(np.float32)
            left_hand_joint_offset_std = left_hand_joint_offset_mean_std["std"].astype(np.float32)
            left_hand_joint_offset_mean = left_hand_joint_offset_mean.reshape(
                (1, 1, left_hand_joint_offset_mean.shape[0], left_hand_joint_offset_mean.shape[1]))
            left_hand_joint_offset_std = left_hand_joint_offset_std.reshape(left_hand_joint_offset_mean.shape)

            left_hand_wrist_marker_pos_mean_std = np.load(left_wrist_marker_pos_mean_std_path)
            left_hand_wrist_marker_pos_mean = left_hand_wrist_marker_pos_mean_std["mean"].astype(np.float32)
            left_hand_wrist_marker_pos_std = left_hand_wrist_marker_pos_mean_std["std"].astype(np.float32)
            left_hand_wrist_marker_pos_mean = left_hand_wrist_marker_pos_mean.reshape(
                (1, 1, *left_hand_wrist_marker_pos_mean.shape))
            left_hand_wrist_marker_pos_std = left_hand_wrist_marker_pos_std.reshape(
                left_hand_wrist_marker_pos_mean.shape)

            left_hand_marker_pos_mean = torch.from_numpy(left_hand_marker_pos_mean).to(device)
            left_hand_marker_pos_std = torch.from_numpy(left_hand_marker_pos_std).to(device)
            left_hand_motion_mean = torch.from_numpy(left_hand_motion_mean).to(device)
            left_hand_motion_std = torch.from_numpy(left_hand_motion_std).to(device)
            left_hand_wrist_marker_pos_mean = torch.from_numpy(left_hand_wrist_marker_pos_mean).to(device)
            left_hand_wrist_marker_pos_std = torch.from_numpy(left_hand_wrist_marker_pos_std).to(device)
            left_hand_joint_offset_mean = torch.from_numpy(left_hand_joint_offset_mean).to(device)
            left_hand_joint_offset_std = torch.from_numpy(left_hand_joint_offset_std).to(device)

            window_left_hand_marker_pos = torch.from_numpy(window_left_hand_marker_pos.astype(np.float32)).to(device)
            window_left_hand_marker_vis = torch.from_numpy(window_left_hand_marker_vis.astype(np.float32)).to(device)
            window_left_wrist_marker_pos = torch.from_numpy(window_left_wrist_marker_pos.astype(np.float32)).to(device)

            normed_left_hand_marker_pos = (window_left_hand_marker_pos - left_hand_marker_pos_mean) / \
                                          left_hand_marker_pos_std
            normed_left_wrist_marker_pos = (window_left_wrist_marker_pos - left_hand_wrist_marker_pos_mean) / \
                                           left_hand_wrist_marker_pos_std

            with torch.no_grad():
                network_input = torch.cat([torch.cat([normed_left_wrist_marker_pos,
                                                      torch.zeros((batch_size, window_shape, 4, 1)).to(device)],
                                                     dim=-1),
                                           torch.cat([normed_left_hand_marker_pos,
                                                      torch.unsqueeze(window_left_hand_marker_vis, dim=-1).repeat(
                                                          (1, 1, 1, 1))], dim=-1)],
                                          dim=2).view((batch_size, window_shape, -1)).float()
                pred = left_hand_network(network_input)
                pred = pred.view((batch_size, window_shape, -1))

            pred_left_hand_motion = torch.cat(
                [pred[:, :, :-3].view((batch_size, window_shape, left_hand_joint_num, -1))[:, :, :, :9].
                 reshape((batch_size, window_shape, -1)), pred[:, :, -3:]], dim=-1)
            pred_left_hand_joint_offset = pred[:, :, :-3].view((batch_size, window_shape, left_hand_joint_num, -1))[:,
                                          :, :, -3:]. \
                reshape((batch_size, window_shape, -1, 3))

            pred_mean_left_hand_joint_offset = torch.mean(pred_left_hand_joint_offset,
                                                          dim=[0,
                                                               1]) * left_hand_joint_offset_std + left_hand_joint_offset_mean
            pred_mean_left_hand_joint_offset = pred_mean_left_hand_joint_offset[0, 0].cpu().numpy()
            pred_mean_left_hand_joint_offset[0] *= 0

            pred_left_hand_motion = pred_left_hand_motion * left_hand_motion_std + left_hand_motion_mean
            cated_pred_left_hand_motion = cat_window_arr(pred_left_hand_motion)
            cated_pred_left_hand_rot_mat = cated_pred_left_hand_motion[:, :-3].reshape(
                (cated_pred_left_hand_motion.shape[0], -1, 3, 3))
            cated_pred_left_hand_trans = cated_pred_left_hand_motion[:, -3:].cpu().numpy()

            ortho_pred_rot = cated_pred_left_hand_rot_mat.clone().detach()

            ortho_pred_rot[..., 1] -= \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 1]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 0])). \
                    view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 0]
            ortho_pred_rot[..., 2] -= \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 2]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 0])). \
                    view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 0] + \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 1], ortho_pred_rot[..., 2]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 1], ortho_pred_rot[..., 1])). \
                    view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 1]
            ortho_pred_rot[..., 0] /= torch.linalg.norm(ortho_pred_rot[..., 0], dim=-1, keepdim=True)
            ortho_pred_rot[..., 1] /= torch.linalg.norm(ortho_pred_rot[..., 1], dim=-1, keepdim=True)
            ortho_pred_rot[..., 2] /= torch.linalg.norm(ortho_pred_rot[..., 2], dim=-1, keepdim=True)

            left_hand_rot_mat = ortho_pred_rot.cpu().numpy()
            left_hand_joint_offset = pred_mean_left_hand_joint_offset

            args.network_dim[0] = (hand_marker_num + 4) * args.marker_gcn_dim[-1]
            args.network_dim[-1] = (right_hand_joint_num + 1) * args.skeleton_gcn_dim[0]
            args.skeleton_gcn_dim[-1] = 9 + 3
            args.marker_num = hand_marker_num + 4
            args.joint_num = right_hand_joint_num
            args.parent_list = right_hand_parent_list
            args.joint_name_list = right_hand_parent_list

            right_hand_network = GraphResNetModel(args, args.hand_connect_mat_path,
                                                  args.ten_finger_right_hand_weight_mean_std_path)

            right_hand_checkpoint = torch.load(right_hand_network_resume_file_path)
            right_hand_network.load_state_dict(right_hand_checkpoint)
            right_hand_network = right_hand_network.to(device)
            right_hand_network.eval()

            right_hand_motion_mean_std = np.load(right_hand_motion_mean_std_path)
            right_hand_motion_mean = right_hand_motion_mean_std["mean"].astype(np.float32)
            right_hand_motion_std = right_hand_motion_mean_std["std"].astype(np.float32)
            right_hand_motion_mean = right_hand_motion_mean.reshape((1, 1, right_hand_motion_mean.shape[0])).astype(
                np.float32)
            right_hand_motion_std = right_hand_motion_std.reshape(right_hand_motion_mean.shape).astype(np.float32)

            right_hand_marker_pos_mean_std = np.load(right_hand_marker_pos_mean_std_path)
            right_hand_marker_pos_mean = right_hand_marker_pos_mean_std["mean"]
            right_hand_marker_pos_std = right_hand_marker_pos_mean_std["std"]
            right_hand_marker_pos_mean = right_hand_marker_pos_mean. \
                reshape((1, 1, right_hand_marker_pos_mean.shape[0], right_hand_marker_pos_mean.shape[1])).astype(
                np.float32)
            right_hand_marker_pos_std = right_hand_marker_pos_std.reshape(right_hand_marker_pos_mean.shape).astype(
                np.float32)

            right_hand_joint_offset_mean_std = np.load(right_hand_joint_offset_mean_std_path)
            right_hand_joint_offset_mean = right_hand_joint_offset_mean_std["mean"].astype(np.float32)
            right_hand_joint_offset_std = right_hand_joint_offset_mean_std["std"].astype(np.float32)
            right_hand_joint_offset_mean = right_hand_joint_offset_mean.reshape(
                (1, 1, right_hand_joint_offset_mean.shape[0], right_hand_joint_offset_mean.shape[1]))
            right_hand_joint_offset_std = right_hand_joint_offset_std.reshape(right_hand_joint_offset_mean.shape)

            right_hand_wrist_marker_pos_mean_std = np.load(right_wrist_marker_pos_mean_std_path)
            right_hand_wrist_marker_pos_mean = right_hand_wrist_marker_pos_mean_std["mean"].astype(np.float32)
            right_hand_wrist_marker_pos_std = right_hand_wrist_marker_pos_mean_std["std"].astype(np.float32)
            right_hand_wrist_marker_pos_mean = right_hand_wrist_marker_pos_mean.reshape(
                (1, 1, *right_hand_wrist_marker_pos_mean.shape))
            right_hand_wrist_marker_pos_std = right_hand_wrist_marker_pos_std.reshape(
                right_hand_wrist_marker_pos_mean.shape)

            right_hand_marker_pos_mean = torch.from_numpy(right_hand_marker_pos_mean).to(device)
            right_hand_marker_pos_std = torch.from_numpy(right_hand_marker_pos_std).to(device)
            right_hand_motion_mean = torch.from_numpy(right_hand_motion_mean).to(device)
            right_hand_motion_std = torch.from_numpy(right_hand_motion_std).to(device)
            right_hand_wrist_marker_pos_mean = torch.from_numpy(right_hand_wrist_marker_pos_mean).to(device)
            right_hand_wrist_marker_pos_std = torch.from_numpy(right_hand_wrist_marker_pos_std).to(device)
            right_hand_joint_offset_mean = torch.from_numpy(right_hand_joint_offset_mean).to(device)
            right_hand_joint_offset_std = torch.from_numpy(right_hand_joint_offset_std).to(device)

            window_right_hand_marker_pos = torch.from_numpy(window_right_hand_marker_pos.astype(np.float32)).to(device)
            window_right_hand_marker_vis = torch.from_numpy(window_right_hand_marker_vis.astype(np.float32)).to(device)
            window_right_wrist_marker_pos = torch.from_numpy(window_right_wrist_marker_pos.astype(np.float32)).to(
                device)

            normed_right_hand_marker_pos = (window_right_hand_marker_pos - right_hand_marker_pos_mean) / \
                                           right_hand_marker_pos_std
            normed_right_wrist_marker_pos = (window_right_wrist_marker_pos - right_hand_wrist_marker_pos_mean) / \
                                            right_hand_wrist_marker_pos_std

            with torch.no_grad():
                network_input = torch.cat([torch.cat([normed_right_wrist_marker_pos,
                                                      torch.zeros((batch_size, window_shape, 4, 1)).to(device)],
                                                     dim=-1),
                                           torch.cat([normed_right_hand_marker_pos,
                                                      torch.unsqueeze(window_right_hand_marker_vis, dim=-1).repeat(
                                                          (1, 1, 1, 1))], dim=-1)],
                                          dim=2).view((batch_size, window_shape, -1)).float()
                pred = right_hand_network(network_input)
                pred = pred.view((batch_size, window_shape, -1))

            pred_right_hand_motion = torch.cat(
                [pred[:, :, :-3].view((batch_size, window_shape, right_hand_joint_num, -1))[:, :, :, :9].
                 reshape((batch_size, window_shape, -1)), pred[:, :, -3:]], dim=-1)
            pred_right_hand_joint_offset = pred[:, :, :-3].view((batch_size, window_shape, right_hand_joint_num, -1))[:,
                                           :, :, -3:]. \
                reshape((batch_size, window_shape, -1, 3))

            pred_mean_right_hand_joint_offset = torch.mean(pred_right_hand_joint_offset,
                                                           dim=[0,
                                                                1]) * right_hand_joint_offset_std + right_hand_joint_offset_mean
            pred_mean_right_hand_joint_offset = pred_mean_right_hand_joint_offset[0, 0].cpu().numpy()
            pred_mean_right_hand_joint_offset[0] *= 0

            pred_right_hand_motion = pred_right_hand_motion * right_hand_motion_std + right_hand_motion_mean
            cated_pred_right_hand_motion = cat_window_arr(pred_right_hand_motion)
            cated_pred_right_hand_rot_mat = cated_pred_right_hand_motion[:, :-3].reshape(
                (cated_pred_right_hand_motion.shape[0], -1, 3, 3))
            cated_pred_right_hand_trans = cated_pred_right_hand_motion[:, -3:].cpu().numpy()

            ortho_pred_rot = cated_pred_right_hand_rot_mat.clone().detach()

            ortho_pred_rot[..., 1] -= \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 1]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 0])). \
                    view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 0]
            ortho_pred_rot[..., 2] -= \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 2]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 0], ortho_pred_rot[..., 0])). \
                    view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 0] + \
                (torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 1], ortho_pred_rot[..., 2]) /
                 torch.einsum("bcd,bcd->bc", ortho_pred_rot[..., 1], ortho_pred_rot[..., 1])). \
                    view((ortho_pred_rot.shape[0], -1, 1)) * ortho_pred_rot[..., 1]
            ortho_pred_rot[..., 0] /= torch.linalg.norm(ortho_pred_rot[..., 0], dim=-1, keepdim=True)
            ortho_pred_rot[..., 1] /= torch.linalg.norm(ortho_pred_rot[..., 1], dim=-1, keepdim=True)
            ortho_pred_rot[..., 2] /= torch.linalg.norm(ortho_pred_rot[..., 2], dim=-1, keepdim=True)

            right_hand_rot_mat = ortho_pred_rot.cpu().numpy()
            right_hand_joint_offset = pred_mean_right_hand_joint_offset

        with open(os.path.join(args.root_dir,
                               args.full_joint_name_config_path), "r") as f:
            content = f.readlines()
            joint_list = [line.split()[0] for line in content]
            parent_list = [int(line.split()[1]) for line in content]
            joint_name_list = joint_list
        all_joint_num = len(joint_list)
        all_joint_name_list = joint_name_list
        all_parent_list = parent_list

        export_to_bvh(out_file, body_joint_name_list, body_rot_mat,
                      pred_body_trans, pred_body_t_pose_joint_pos, np.asarray(body_parent_list))

        all_rot_mat = np.zeros((body_rot_mat.shape[0], all_joint_num, 3, 3))
        all_joint_offset = np.zeros((all_joint_num, 3))
        all_joint_t_pose_pos = np.zeros((all_joint_num, 3))

        for ji, joint_name in enumerate(body_joint_name_list):
            idx = all_joint_name_list.index(joint_name)
            all_rot_mat[:, idx] = body_rot_mat[:, ji]
            all_joint_offset[idx] = body_joint_offset[ji]

        for ji, joint_name in enumerate(left_hand_joint_name_list):
            if ji == 0:
                continue
            idx = all_joint_name_list.index(joint_name)
            all_rot_mat[:, idx] = left_hand_rot_mat[:, ji]
            all_joint_offset[idx] = left_hand_joint_offset[ji]

        for ji, joint_name in enumerate(right_hand_joint_name_list):
            if ji == 0:
                continue
            idx = all_joint_name_list.index(joint_name)
            all_rot_mat[:, idx] = right_hand_rot_mat[:, ji]
            all_joint_offset[idx] = right_hand_joint_offset[ji]

        for i in range(1, len(all_joint_name_list)):
            all_joint_t_pose_pos[i] = all_joint_t_pose_pos[all_parent_list[i]] + \
                                      all_joint_offset[i]

        # np.save("out.npy", marker_pos)
        export_to_bvh(out_file, all_joint_name_list, all_rot_mat,
                      pred_body_trans, all_joint_t_pose_pos, np.asarray(all_parent_list))


if __name__ == "__main__":
    pass

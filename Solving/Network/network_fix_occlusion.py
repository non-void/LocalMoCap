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

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from .models import *
from .utils.average_meter import AverageMeter
from .utils.marker_utils import *

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
    out = np.zeros((anim_length, arr.shape[2], arr.shape[3]))
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


def network_fix_occlusion(args, no_fix_file, intermediate_file, out_file, verbose=True):
    npz_content = dict(np.load(intermediate_file))

    marker_names = marker_name_config["body"]["front_waist"] + marker_name_config["finger"]["10"]
    marker_vis = npz_content["corrupted_marker_vis"]
    marker_pos = npz_content["traditional_fixed_marker"]

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
            window_right_hand_marker_vis = \
            batch_generate_window_data([aligned_body_marker_pos,
                                        aligned_left_hand_marker_pos,
                                        aligned_right_hand_marker_pos,
                                        aligned_left_wrist_marker_pos,
                                        aligned_right_wrist_marker_pos,
                                        body_marker_vis,
                                        left_hand_marker_vis,
                                        right_hand_marker_vis],
                                       WINDOW_SIZE)

        ## Networks
        network_fixed_character_marker_pos = copy.deepcopy(character_marker_pos)

        if args.cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        if character_config["body"] == "front_waist":
            body_marker_pos_mean_std_path = args.front_waist_body_marker_pos_mean_std_path
            body_network_resume_file_path = args.front_waist_body_resume_file_path

        body_network = GNNBiLSTM(args, body_marker_num, body_marker_num * 6,
                                 args.body_hidden_dim, body_marker_num * 3,
                                 args.body_connect_mat_path, layer_num=args.lstm_layer_num)

        body_checkpoint = torch.load(body_network_resume_file_path)
        body_network.load_state_dict(body_checkpoint)
        body_network = body_network.to(device)
        body_network.eval()

        body_marker_pos_mean_std = np.load(body_marker_pos_mean_std_path)
        body_marker_pos_mean = body_marker_pos_mean_std["mean"]
        body_marker_pos_std = body_marker_pos_mean_std["std"]
        body_marker_pos_mean = body_marker_pos_mean.reshape((1, 1, *body_marker_pos_mean.shape))
        body_marker_pos_std = body_marker_pos_std.reshape(body_marker_pos_mean.shape)
        body_marker_pos_mean = torch.from_numpy(body_marker_pos_mean).to(device)
        body_marker_pos_std = torch.from_numpy(body_marker_pos_std).to(device)

        window_body_marker_pos = torch.from_numpy(window_body_marker_pos.astype(np.float32)).to(device)

        batch_size, window_shape, marker_num = \
            window_body_marker_pos.shape[0], \
                window_body_marker_pos.shape[1], \
                window_body_marker_pos.shape[2]

        normed_body_marker_pos = \
            (window_body_marker_pos - body_marker_pos_mean) / body_marker_pos_std
        window_body_marker_vis_tensor = torch.from_numpy(window_body_marker_vis).to(device)

        with torch.no_grad():
            network_input = torch.cat([normed_body_marker_pos,
                                       torch.unsqueeze(window_body_marker_vis_tensor, -1).repeat((1, 1, 1, 3))],
                                      dim=3).float()
            pred, _ = body_network(network_input)
            pred = pred.view(normed_body_marker_pos.shape).detach().cpu().numpy()

        window_body_marker_pos = window_body_marker_pos.detach().cpu().numpy()

        out_body_marker_pos = \
            (1 - np.expand_dims(window_body_marker_vis, -1)) * window_body_marker_pos + \
            np.expand_dims(window_body_marker_vis, -1) * (window_body_marker_pos + pred)

        network_fixed_aligned_body_marker_pos = cat_window_arr(out_body_marker_pos)[:anim_length]
        network_fixed_body_marker_pos = \
            add_marker_global_transform(network_fixed_aligned_body_marker_pos,
                                        global_body_rot_mat,
                                        global_body_trans)
        network_fixed_character_marker_pos[:, body_marker_index] = network_fixed_body_marker_pos

        if have_hand:
            if hand_marker_num == 10:
                left_hand_resume_file_path = args.ten_finger_left_hand_resume_file_path
                left_wrist_marker_pos_mean_std_path = args.ten_finger_left_wrist_marker_pos_mean_std_path
                left_hand_marker_pos_mean_std_path = args.ten_finger_left_hand_marker_pos_mean_std_path

                right_hand_resume_file_path = args.ten_finger_right_hand_resume_file_path
                right_wrist_marker_pos_mean_std_path = args.ten_finger_right_wrist_marker_pos_mean_std_path
                right_hand_marker_pos_mean_std_path = args.ten_finger_right_hand_marker_pos_mean_std_path

            left_hand_network = GNNBiLSTM(args, hand_marker_num + 4, (hand_marker_num + 4) * 3 + hand_marker_num * 3,
                                          args.hand_hidden_dim, hand_marker_num * 3,
                                          args.hand_connect_mat_path, act="Mish", layer_num=args.lstm_layer_num)
            left_hand_checkpoint = torch.load(left_hand_resume_file_path)
            left_hand_network.load_state_dict(left_hand_checkpoint)
            left_hand_network = left_hand_network.to(device)
            left_hand_network.eval()

            left_wrist_marker_pos_mean_std = np.load(left_wrist_marker_pos_mean_std_path)
            left_wrist_marker_pos_mean = left_wrist_marker_pos_mean_std["mean"]
            left_wrist_marker_pos_std = left_wrist_marker_pos_mean_std["std"]
            left_wrist_marker_pos_mean = left_wrist_marker_pos_mean.reshape((1, 1, *left_wrist_marker_pos_mean.shape))
            left_wrist_marker_pos_std = left_wrist_marker_pos_std.reshape(left_wrist_marker_pos_mean.shape)
            left_wrist_marker_pos_mean = torch.from_numpy(left_wrist_marker_pos_mean).to(device)
            left_wrist_marker_pos_std = torch.from_numpy(left_wrist_marker_pos_std).to(device)

            left_hand_marker_pos_mean_std = np.load(left_hand_marker_pos_mean_std_path)
            left_hand_marker_pos_mean = left_hand_marker_pos_mean_std["mean"]
            left_hand_marker_pos_std = left_hand_marker_pos_mean_std["std"]
            left_hand_marker_pos_mean = left_hand_marker_pos_mean.reshape((1, 1, *left_hand_marker_pos_mean.shape))
            left_hand_marker_pos_std = left_hand_marker_pos_std.reshape(left_hand_marker_pos_mean.shape)
            left_hand_marker_pos_mean = torch.from_numpy(left_hand_marker_pos_mean).to(device)
            left_hand_marker_pos_std = torch.from_numpy(left_hand_marker_pos_std).to(device)

            window_left_hand_marker_pos = torch.from_numpy(window_left_hand_marker_pos.astype(np.float32)).to(device)
            window_left_wrist_marker_pos = torch.from_numpy(window_left_wrist_marker_pos.astype(np.float32)).to(device)

            batch_size, window_shape, marker_num = \
                window_left_hand_marker_pos.shape[0], \
                    window_left_hand_marker_pos.shape[1], \
                    window_left_hand_marker_pos.shape[2]

            normed_left_hand_marker_pos = \
                (window_left_hand_marker_pos - left_hand_marker_pos_mean) / left_hand_marker_pos_std
            normed_left_wrist_marker_pos = \
                (window_left_wrist_marker_pos - left_wrist_marker_pos_mean) / left_wrist_marker_pos_std
            window_left_hand_marker_vis_tensor = torch.from_numpy(window_left_hand_marker_vis).to(device)

            with torch.no_grad():
                pred, _ = left_hand_network(
                    torch.cat([torch.cat([normed_left_wrist_marker_pos,
                                          torch.zeros((batch_size, window_shape, 4, 3)).to(device)], dim=-1),
                               torch.cat([normed_left_hand_marker_pos,
                                          torch.unsqueeze(window_left_hand_marker_vis_tensor, dim=-1).repeat(
                                              (1, 1, 1, 3))], dim=-1)],
                              dim=2).view((batch_size, window_shape, -1)).float())
                pred = pred.view(window_left_hand_marker_pos.shape).detach().cpu().numpy()

            window_left_hand_marker_pos = window_left_hand_marker_pos.detach().cpu().numpy()

            out_left_hand_marker_pos = \
                (1 - np.expand_dims(window_left_hand_marker_vis, -1)) * window_left_hand_marker_pos + \
                np.expand_dims(window_left_hand_marker_vis, -1) * (window_left_hand_marker_pos + pred)

            network_fixed_aligned_left_hand_marker_pos = cat_window_arr(out_left_hand_marker_pos)[:anim_length]
            network_fixed_left_hand_marker_pos = \
                add_marker_global_transform(network_fixed_aligned_left_hand_marker_pos,
                                            global_left_hand_rot_mat,
                                            global_left_hand_trans)

            right_hand_network = GNNBiLSTM(args, hand_marker_num + 4, (hand_marker_num + 4) * 3 + hand_marker_num * 3,
                                           args.hand_hidden_dim, hand_marker_num * 3,
                                           args.hand_connect_mat_path, act="Mish", layer_num=args.lstm_layer_num)
            right_hand_checkpoint = torch.load(right_hand_resume_file_path)
            right_hand_network.load_state_dict(right_hand_checkpoint)
            right_hand_network = right_hand_network.to(device)
            right_hand_network.eval()

            right_wrist_marker_pos_mean_std = np.load(right_wrist_marker_pos_mean_std_path)
            right_wrist_marker_pos_mean = right_wrist_marker_pos_mean_std["mean"]
            right_wrist_marker_pos_std = right_wrist_marker_pos_mean_std["std"]
            right_wrist_marker_pos_mean = right_wrist_marker_pos_mean.reshape(
                (1, 1, *right_wrist_marker_pos_mean.shape))
            right_wrist_marker_pos_std = right_wrist_marker_pos_std.reshape(right_wrist_marker_pos_mean.shape)
            right_wrist_marker_pos_mean = torch.from_numpy(right_wrist_marker_pos_mean).to(device)
            right_wrist_marker_pos_std = torch.from_numpy(right_wrist_marker_pos_std).to(device)

            right_hand_marker_pos_mean_std = np.load(right_hand_marker_pos_mean_std_path)
            right_hand_marker_pos_mean = right_hand_marker_pos_mean_std["mean"]
            right_hand_marker_pos_std = right_hand_marker_pos_mean_std["std"]
            right_hand_marker_pos_mean = right_hand_marker_pos_mean.reshape((1, 1, *right_hand_marker_pos_mean.shape))
            right_hand_marker_pos_std = right_hand_marker_pos_std.reshape(right_hand_marker_pos_mean.shape)
            right_hand_marker_pos_mean = torch.from_numpy(right_hand_marker_pos_mean).to(device)
            right_hand_marker_pos_std = torch.from_numpy(right_hand_marker_pos_std).to(device)

            window_right_hand_marker_pos = torch.from_numpy(window_right_hand_marker_pos.astype(np.float32)).to(device)
            window_right_wrist_marker_pos = torch.from_numpy(window_right_wrist_marker_pos.astype(np.float32)).to(
                device)

            batch_size, window_shape, marker_num = \
                window_right_hand_marker_pos.shape[0], \
                    window_right_hand_marker_pos.shape[1], \
                    window_right_hand_marker_pos.shape[2]

            normed_right_hand_marker_pos = \
                (window_right_hand_marker_pos - right_hand_marker_pos_mean) / right_hand_marker_pos_std
            normed_right_wrist_marker_pos = \
                (window_right_wrist_marker_pos - right_wrist_marker_pos_mean) / right_wrist_marker_pos_std
            window_right_hand_marker_vis_tensor = torch.from_numpy(window_right_hand_marker_vis).to(device)

            with torch.no_grad():
                pred, _ = right_hand_network(
                    torch.cat([torch.cat([normed_right_wrist_marker_pos,
                                          torch.zeros((batch_size, window_shape, 4, 3)).to(device)], dim=-1),
                               torch.cat([normed_right_hand_marker_pos,
                                          torch.unsqueeze(window_right_hand_marker_vis_tensor, dim=-1).repeat(
                                              (1, 1, 1, 3))], dim=-1)],
                              dim=2).view((batch_size, window_shape, -1)).float())
                pred = pred.view(window_right_hand_marker_pos.shape).detach().cpu().numpy()

            window_right_hand_marker_pos = window_right_hand_marker_pos.detach().cpu().numpy()

            out_right_hand_marker_pos = \
                (1 - np.expand_dims(window_right_hand_marker_vis, -1)) * window_right_hand_marker_pos + \
                np.expand_dims(window_right_hand_marker_vis, -1) * (window_right_hand_marker_pos + pred)

            network_fixed_aligned_right_hand_marker_pos = cat_window_arr(out_right_hand_marker_pos)[:anim_length]
            network_fixed_right_hand_marker_pos = \
                add_marker_global_transform(network_fixed_aligned_right_hand_marker_pos,
                                            global_right_hand_rot_mat,
                                            global_right_hand_trans)

            network_fixed_character_marker_pos[:, left_hand_marker_index] = network_fixed_left_hand_marker_pos
            network_fixed_character_marker_pos[:, right_hand_marker_index] = network_fixed_right_hand_marker_pos

    npz_content["network_fixed_marker_pos"] = network_fixed_character_marker_pos
    np.savez(out_file, **npz_content)


if __name__ == "__main__":
    pass

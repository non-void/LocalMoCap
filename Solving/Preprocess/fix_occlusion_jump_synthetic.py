import json
import os
import time
import copy

import scipy.signal
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt

from .OutlierRemoval.PREEEDM import procrustes_zhou
from .c3d_utils import *


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


def find_invis_markers(marker_pos, c3d_vis):
    # it seems that in c3d files, the nan marker pos exactly matches the c3d invis array,
    # thus there is no need check both array
    anim_length = marker_pos.shape[0]
    marker_num = marker_pos.shape[1]
    invis_marker_pos = np.zeros((anim_length, marker_num))
    invis_marker_pos2 = np.zeros((anim_length, marker_num))

    marker_pos_nan_location = np.where(np.isnan(marker_pos))
    invis_marker_pos[marker_pos_nan_location[0], marker_pos_nan_location[1]] = 1
    invis_marker_pos2[np.where(c3d_vis[0].transpose())] = 1

    assert np.all(invis_marker_pos == invis_marker_pos2)
    return np.where(invis_marker_pos)


def calc_dist_matrix(marker_pos):
    anim_length = marker_pos.shape[0]
    marker_num = marker_pos.shape[1]

    dist_matrix_arr = np.zeros((anim_length, marker_num, marker_num))
    for i in range(anim_length):
        dist_matrix_arr[i] = squareform(pdist(marker_pos[i]))
    return dist_matrix_arr


def find_ref_frame(frame, marker_vis_arr):
    anim_length = marker_vis_arr.shape[0]
    marker_vis = marker_vis_arr[frame]
    invis_marker_list = np.where(marker_vis)[0]
    ref_frame_list = np.zeros((2, invis_marker_list.shape[0]), dtype=int)
    for ii, invis_marker_idx in enumerate(invis_marker_list):
        back = frame - 1
        while back != -1:
            if not marker_vis_arr[back, invis_marker_idx]:
                break
            back -= 1
        ref_frame_list[0, ii] = back

        forward = frame + 1
        while forward != anim_length:
            if not marker_vis_arr[forward, invis_marker_idx]:
                break
            forward += 1
        ref_frame_list[1, ii] = forward
        if back == -1 and forward == anim_length:
            raise Exception("Marker {} invisible all the time!")

    return invis_marker_list, ref_frame_list


def fix_marker_pos_per_frame(args, frame, marker_pos_arr, marker_vis_arr, invis_marker_list, ref_frame_list,
                             dist_matrix_list, dist_matrix_std, dist_matrix_mean):
    marker_num = marker_pos_arr.shape[1]
    fixed_marker_pos = copy.deepcopy(marker_pos_arr[frame])
    anim_length = marker_pos_arr.shape[0]

    ref_marker_num = args.fix_occlusion_ref_marker_num

    for ii, invis_marker_idx in enumerate(invis_marker_list):
        marker_pos = marker_pos_arr[frame]
        # ref_frame = ref_frame_list[ii]
        back_ref_frame = ref_frame_list[0, ii]
        forward_ref_frame = ref_frame_list[1, ii]
        if back_ref_frame == -1 and forward_ref_frame != anim_length:
            visible_marker_list = np.setdiff1d(np.arange(marker_num),
                                               np.hstack([np.where(marker_vis_arr[forward_ref_frame])[0],
                                                          invis_marker_list]))
            dist_matrix = dist_matrix_list[forward_ref_frame]

        elif back_ref_frame != -1 and forward_ref_frame == anim_length:
            visible_marker_list = np.setdiff1d(np.arange(marker_num),
                                               np.hstack([np.where(marker_vis_arr[back_ref_frame])[0],
                                                          invis_marker_list]))
            dist_matrix = dist_matrix_list[back_ref_frame]

        elif back_ref_frame != -1 and forward_ref_frame != anim_length:
            visible_marker_list = np.setdiff1d(np.arange(marker_num),
                                               np.hstack([np.where(marker_vis_arr[back_ref_frame])[0],
                                                          np.where(marker_vis_arr[forward_ref_frame])[0],
                                                          invis_marker_list]))
            dist_matrix = ((frame - back_ref_frame) / (forward_ref_frame - back_ref_frame)) * \
                          dist_matrix_list[forward_ref_frame] + \
                          ((forward_ref_frame - frame) / (forward_ref_frame - back_ref_frame)) * \
                          dist_matrix_list[back_ref_frame]

        marker_dist_std = dist_matrix_std[:, invis_marker_idx]
        marker_dist_mean = dist_matrix_mean[:, invis_marker_idx]

        if args.fix_occlusion_ref_marker_method == "std":
            visible_marker_dist_std_with_idx = np.vstack((np.arange(marker_num).reshape(1, -1),
                                                          marker_dist_std.reshape((1, -1))))[:, visible_marker_list]
            ref_marker_idx = visible_marker_dist_std_with_idx[
                0, np.lexsort(visible_marker_dist_std_with_idx)[:ref_marker_num]].astype(int)
        elif args.fix_occlusion_ref_marker_method == "mean":
            visible_marker_dist_mean_with_idx = np.vstack((np.arange(marker_num).reshape(1, -1),
                                                           marker_dist_mean.reshape((1, -1))))[:, visible_marker_list]
            ref_marker_idx = visible_marker_dist_mean_with_idx[
                0, np.lexsort(visible_marker_dist_mean_with_idx)[:ref_marker_num]].astype(int)

        extended_marker_idx = np.hstack([ref_marker_idx, [invis_marker_idx]])
        ref_marker_pos = marker_pos[ref_marker_idx]

        sub_dist_matrix = dist_matrix[extended_marker_idx][:, extended_marker_idx]
        try:
            invis_marker_pos = procrustes_zhou(ref_marker_num, ref_marker_pos.transpose(), sub_dist_matrix ** 2)
            fixed_marker_pos[invis_marker_idx] = invis_marker_pos[:, 0]
        except Exception:
            forward_ref_frame = min(anim_length - 1, forward_ref_frame)
            fixed_marker_pos[invis_marker_idx] = \
                ((frame - back_ref_frame) / (forward_ref_frame - back_ref_frame)) * \
                marker_pos_arr[forward_ref_frame, invis_marker_idx] + \
                ((forward_ref_frame - frame) / (forward_ref_frame - back_ref_frame)) * \
                marker_pos_arr[back_ref_frame, invis_marker_idx]

    return fixed_marker_pos


def fix_marker_occlusion(args, character_marker_pos, character_marker_vis, verbose=True):
    dist_matrix_list = calc_dist_matrix(character_marker_pos)
    dist_matrix_std = np.nanstd(dist_matrix_list, axis=0)
    dist_matrix_mean = np.nanmean(dist_matrix_list, axis=0)

    anim_length = character_marker_pos.shape[0]
    fixed_character_marker_pos = copy.deepcopy(character_marker_pos)
    fix_marker_num = np.where(character_marker_vis)[0].shape[0]
    start_time = time.time()
    for frame in range(anim_length):
        if np.any(character_marker_vis[frame]):
            invis_marker_list, ref_frame_list = find_ref_frame(frame, character_marker_vis)
            # print(frame, invis_marker_list, ref_frame_list)
            fixed_marker_pos = fix_marker_pos_per_frame(args, frame, character_marker_pos,
                                                        character_marker_vis,
                                                        invis_marker_list, ref_frame_list,
                                                        dist_matrix_list, dist_matrix_std, dist_matrix_mean)
            fixed_character_marker_pos[frame] = fixed_marker_pos
    # smoothed_character_marker_pos = np.zeros_like(fixed_character_marker_pos)
    # for i in range(character_marker_pos.shape[1]):
    #     for j in range(character_marker_pos.shape[2]):
    #         smoothed_character_marker_pos[:, i, j] = savgol_filter(fixed_character_marker_pos[:, i, j],
    #                                                                window_length=11, polyorder=3)
    # fixed_character_marker_pos[np.where(character_marker_vis)] = \
    #     smoothed_character_marker_pos[np.where(character_marker_vis)]
    end_time = time.time()
    if verbose:
        print("cost {:.4f}s, fixed {} markers".format(end_time - start_time, fix_marker_num))

    return fixed_character_marker_pos


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


def plot_peak(data, peaks, peaks_property=None, title=None):
    plt.plot(data)
    plt.plot(peaks, data[peaks], "xr")
    if title:
        plt.title(title)

    if peaks_property:
        for ii, peak_loc in enumerate(peaks):
            print(ii, peak_loc, "prominence", peaks_property["prominences"][ii],
                  "left base", peaks_property["left_bases"][ii],
                  "right base", peaks_property["right_bases"][ii])
    plt.show()


def fix_full_body_marker_jump(args, character_name, character_config, character_marker_pos,
                              character_marker_name, verbose=True):
    PRODUCTION_WAIST_MARKER_NAMES = ["LFWT", "LMWT", "LBWT", "RFWT", "RMWT", "RBWT", "STRN", "T10"]
    FRONT_WAIST_MARKER_NAMES = ["LFWT", "MFWT", "RFWT", "LBWT", "MBWT", "RBWT", "STRN", "T10"]
    LEFT_WRIST_MARKER_NAMES = ["LIWR", "LOWR", "LIHAND", "LOHAND"]
    RIGHT_WRIST_MARKER_NAMES = ["RIWR", "ROWR", "RIHAND", "ROHAND"]
    PRODUCTION_WAIST_MARKER_FILE = "configs/production_up_waist.npy"
    FRONT_WAIST_WAIST_MARKER_FILE = "configs/front_up_waist.npy"
    production_waist_marker_pos = np.load(PRODUCTION_WAIST_MARKER_FILE)
    front_waist_marker_pos = np.load(FRONT_WAIST_WAIST_MARKER_FILE)
    production_waist_marker_pos -= np.mean(production_waist_marker_pos, axis=0)
    front_waist_marker_pos -= np.mean(front_waist_marker_pos, axis=0)

    if character_config["body"] == "production":
        waist_marker_names = PRODUCTION_WAIST_MARKER_NAMES
        ref_waist_marker_pos = production_waist_marker_pos
    elif character_config["body"] == "front_waist":
        waist_marker_names = FRONT_WAIST_MARKER_NAMES
        ref_waist_marker_pos = front_waist_marker_pos
    else:
        raise Exception("Unknown body marker config")

    body_marker_names = marker_name_config["body"][character_config["body"]]
    hand_marker_names = marker_name_config["finger"][character_config["finger"]]
    left_hand_marker_names = list(filter(lambda name: name[0] == "L",
                                         hand_marker_names))
    right_hand_marker_names = list(filter(lambda name: name[0] == "R",
                                          hand_marker_names))

    character_marker_name_no_prefix = [name[name.find(":") + 1:] for name in character_marker_name]
    waist_marker_index = [character_marker_name_no_prefix.index(marker_name) \
                          for marker_name in waist_marker_names]
    body_marker_index = [character_marker_name_no_prefix.index(marker_name) \
                         for marker_name in body_marker_names]
    left_hand_marker_index = [character_marker_name_no_prefix.index(marker_name) \
                              for marker_name in left_hand_marker_names]
    right_hand_marker_index = [character_marker_name_no_prefix.index(marker_name) \
                               for marker_name in right_hand_marker_names]
    left_wrist_marker_index = [character_marker_name_no_prefix.index(marker_name) \
                               for marker_name in LEFT_WRIST_MARKER_NAMES]
    right_wrist_marker_index = [character_marker_name_no_prefix.index(marker_name) \
                                for marker_name in RIGHT_WRIST_MARKER_NAMES]

    fixed_character_marker_pos = copy.deepcopy(character_marker_pos)
    if verbose:
        print("Fixing waist marker jumps")
    start_time = time.time()

    waist_marker_pos = character_marker_pos[:, waist_marker_index]
    jump_fixed_waist_marker_pos = fix_jump(args, waist_marker_pos,
                                           args.fix_waist_jump_peak_threshold,
                                           args.fix_waist_jump_peak_interval,
                                           args.fix_waist_jump_loop_num)
    fixed_character_marker_pos[:, waist_marker_index] = jump_fixed_waist_marker_pos

    end_time = time.time()
    if verbose:
        print("cost {:.4f}s".format(end_time - start_time))

    if verbose:
        print("Fixing body marker jumps")
    start_time = time.time()

    body_marker_pos = fixed_character_marker_pos[:, body_marker_index]
    waist_marker_pos = fixed_character_marker_pos[:, waist_marker_index]
    _, global_body_rot_mat, global_body_trans = align_body_markers(body_marker_pos,
                                                                   waist_marker_pos,
                                                                   ref_waist_marker_pos,
                                                                   character_config["body"])
    aligned_body_marker_pos = remove_marker_global_transform(body_marker_pos,
                                                             global_body_rot_mat,
                                                             global_body_trans)
    jump_fixed_body_marker_pos = fix_jump(args, body_marker_pos,
                                          args.fix_body_jump_peak_threshold,
                                          args.fix_body_jump_peak_interval,
                                          args.fix_body_jump_loop_num)

    # jump_fixed_aligned_body_marker_pos = fix_jump(args, aligned_body_marker_pos,
    #                                               args.fix_body_jump_peak_threshold,
    #                                               args.fix_body_jump_peak_interval,
    #                                               args.fix_body_jump_loop_num)

    # np.save("body.npy", aligned_body_marker_pos)

    # jump_fixed_body_marker_pos = add_marker_global_transform(jump_fixed_aligned_body_marker_pos,
    #                                                          global_body_rot_mat,
    #                                                          global_body_trans)
    fixed_character_marker_pos[:, body_marker_index] = jump_fixed_body_marker_pos

    end_time = time.time()
    if verbose:
        print("cost {:.4f}s".format(end_time - start_time))

    if character_config["finger"] != "0":
        if verbose:
            print("Fixing hand marker jumps")

        start_time = time.time()

        left_hand_marker_pos = fixed_character_marker_pos[:, left_hand_marker_index]
        right_hand_marker_pos = fixed_character_marker_pos[:, right_hand_marker_index]
        left_wrist_marker_pos = fixed_character_marker_pos[:, left_wrist_marker_index]
        right_wrist_marker_pos = fixed_character_marker_pos[:, right_wrist_marker_index]

        _, global_left_hand_rot_mat, global_left_hand_trans = \
            align_hand_markers(left_hand_marker_pos,
                               left_wrist_marker_pos)
        _, global_right_hand_rot_mat, global_right_hand_trans = \
            align_hand_markers(right_hand_marker_pos,
                               right_wrist_marker_pos)

        # aligned_left_hand_marker_pos = remove_marker_global_transform(left_hand_marker_pos,
        #                                                               global_left_hand_rot_mat,
        #                                                               global_left_hand_trans)
        # aligned_right_hand_marker_pos = remove_marker_global_transform(right_hand_marker_pos,
        #                                                                global_right_hand_rot_mat,
        #                                                                global_right_hand_trans)

        # np.save("left_hand.npy", aligned_left_hand_marker_pos)
        # np.save("right_hand.npy", aligned_right_hand_marker_pos)

        jump_fixed_left_hand_marker_pos = fix_jump(args, left_hand_marker_pos,
                                                   args.fix_hand_jump_peak_threshold,
                                                   args.fix_hand_jump_peak_interval,
                                                   args.fix_hand_jump_loop_num)
        jump_fixed_right_hand_marker_pos = fix_jump(args, right_hand_marker_pos,
                                                    args.fix_hand_jump_peak_threshold,
                                                    args.fix_hand_jump_peak_interval,
                                                    args.fix_hand_jump_loop_num)

        # jump_fixed_left_hand_marker_pos = \
        #     add_marker_global_transform(jump_fixed_aligned_left_hand_marker_pos,
        #                                 global_left_hand_rot_mat,
        #                                 global_left_hand_trans)
        # jump_fixed_right_hand_marker_pos = \
        #     add_marker_global_transform(jump_fixed_aligned_right_hand_marker_pos,
        #                                 global_right_hand_rot_mat,
        #                                 global_right_hand_trans)

        fixed_character_marker_pos[:, left_hand_marker_index] = jump_fixed_left_hand_marker_pos
        fixed_character_marker_pos[:, right_hand_marker_index] = jump_fixed_right_hand_marker_pos

        end_time = time.time()
        if verbose:
            print("cost {:.4f}s".format(end_time - start_time))

    return fixed_character_marker_pos


def fix_jump(args, character_marker_pos, peak_threshold, peak_interval, loop_num, verbose=True):
    def get_interpolate_interval(peaks, interpolate_range, anim_length):
        intervals = [[max(0, peak - interpolate_range),
                      min(peak + interpolate_range, anim_length - 1)] for peak in peaks]
        merged_intervals = []

        for i in range(len(intervals)):
            if len(merged_intervals) == 0 or merged_intervals[-1][1] < intervals[i][0]:
                merged_intervals.append(copy.deepcopy(intervals[i]))
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], intervals[i][1])

        return merged_intervals

    anim_length = character_marker_pos.shape[0]
    marker_num = character_marker_pos.shape[1]

    # out_vis_arr = np.zeros((character_marker_pos.shape[0], character_marker_pos.shape[1]))

    fix_jump_marker_pos = copy.deepcopy(character_marker_pos)
    for _ in range(loop_num):
        for marker_idx in range(marker_num):
            marker_pos = character_marker_pos[:, marker_idx]
            marker_vel = marker_pos[1:] - marker_pos[:-1]
            marker_acc = marker_vel[1:] - marker_vel[:-1]
            marker_vel_norm = np.linalg.norm(marker_vel, axis=1)
            marker_acc_norm = np.linalg.norm(marker_acc, axis=1)

            marker_vel_norm = np.concatenate([[marker_vel_norm[0]], marker_vel_norm])
            marker_acc_norm = np.concatenate([[marker_acc_norm[0]], marker_acc_norm, [marker_acc_norm[-1]]])
            marker_curvature = np.abs(marker_acc_norm) / ((1 + marker_vel_norm ** 2) ** (3 / 2))

            peaks, peaks_property = find_peaks(marker_acc_norm, prominence=peak_threshold)

            # if marker_idx == 52:
            #     plot_peak(marker_acc_norm, peaks)

            if peaks.shape[0] != 0:
                interpolate_interval = get_interpolate_interval(peaks, peak_interval, anim_length)
                for interval in interpolate_interval:
                    x = list(range(max(0, interval[0] - peak_interval), interval[0])) + \
                        list(range(interval[1], min(interval[1] + peak_interval, anim_length - 1)))
                    y = fix_jump_marker_pos[x, marker_idx]

                    # for i in range(3):
                    #     fix_jump_marker_pos[interval[0]:interval[1], marker_idx, i] = \
                    #         scipy.signal.savgol_filter(fix_jump_marker_pos[interval[0]:interval[1], marker_idx, i],
                    #                                    13 if interval[1] - interval[0] >= 13 else
                    #                                    (interval[1] - interval[0]) // 2 * 2 - 1, 3)

                    fix_jump_marker_pos[range(interval[0], interval[1]), marker_idx] = \
                        interp1d(x, y.transpose(), kind="slinear", fill_value="extrapolate") \
                            (range(interval[0], interval[1])).transpose()
                # for interval in interpolate_interval:
                #     out_vis_arr[interval[0]:interval[1] + 1, marker_idx] = 1

        # fix_jump_marker_pos = fix_marker_occlusion(args, character_marker_pos, out_vis_arr, verbose=False)
        character_marker_pos = fix_jump_marker_pos

    # for i in range(marker_num):
    #     for j in range(3):
    #         fix_jump_marker_pos[:, i, j] = scipy.signal.savgol_filter(fix_jump_marker_pos[:, i, j], 11, 4)

    return fix_jump_marker_pos


def fix_data_traditional_method(args, in_file_path, out_file_path, fix_occlusion=True, fix_marker_jump=True):
    if not (fix_occlusion or fix_marker_jump):
        raise Exception("No fix configured!")

    npz_content = dict(np.load(in_file_path))
    marker_names = marker_name_config["body"]["front_waist"] + marker_name_config["finger"]["10"]
    marker_vis = npz_content["corrupted_marker_vis"]
    marker_pos = npz_content["marker_corrupted"]

    marker_names = ["character:{}".format(name) for name in marker_names]

    character_list, character_marker_config_list, character_marker_pos_list, \
        character_marker_vis_list, character_marker_index_list, character_marker_name_list = \
        get_character_marker_pos_vis_index(marker_names, marker_pos, marker_vis,
                                           marker_name_dict, marker_name_config)

    for character_idx, character_name in enumerate(character_list):
        if args.verbose:
            print("Processing character \"{}\"".format(character_name))

        if character_idx >= len(character_marker_config_list):
            print('something wrong on character_idx and character_marker_config_list, list length=',
                  len(character_marker_config_list), ' character_idx=', character_idx)
            continue

        character_config = character_marker_config_list[character_idx]
        character_marker_pos = character_marker_pos_list[character_idx]
        character_marker_vis = character_marker_vis_list[character_idx]
        character_marker_name = character_marker_name_list[character_idx]
        character_marker_index = character_marker_index_list[character_idx]

        if fix_occlusion:
            fixed_character_marker_pos = fix_marker_occlusion(args, character_marker_pos,
                                                              character_marker_vis, verbose=args.verbose)
            fixed_dist_matrix_list = calc_dist_matrix(character_marker_pos)
            # fixed_c3d_trans = inverse_transform_c3d_marker_pos(fixed_character_marker_pos)
        else:
            fixed_character_marker_pos = character_marker_pos
        if fix_marker_jump:
            no_jump_character_marker_pos = fix_full_body_marker_jump(args, character_name,
                                                                     character_config,
                                                                     fixed_character_marker_pos,
                                                                     character_marker_name, verbose=args.verbose)
            fixed_character_marker_pos = no_jump_character_marker_pos

    # np.save("no_fix.npy", character_marker_pos)
    # np.save("fixed.npy", fixed_character_marker_pos)
    npz_content["traditional_fixed_marker"] = fixed_character_marker_pos
    np.savez(out_file_path, **npz_content)


MARKER_NAME_CONFIG_PATH = "configs/marker_names.json"

# generate marker config dict to check the given markers belongs to which marker config
marker_name_config = json.load(open(MARKER_NAME_CONFIG_PATH, "r"))
marker_name_dict = dict()
for body_type in marker_name_config["body"].keys():
    for finger_type in marker_name_config["finger"].keys():
        marker_set = frozenset(marker_name_config["body"][body_type]).union(
            frozenset(marker_name_config["finger"][finger_type]))
        marker_name_dict[marker_set] = {"body": body_type, "finger": finger_type}

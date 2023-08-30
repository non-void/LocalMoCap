import json
import os
import numpy as np
from tqdm import tqdm
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
import time
from ezc3d import c3d

from PREEEDM import procrustes_zhou
from c3d_utils import *


def get_character_marker_pos_vis_index(marker_names, marker_pos, marker_vis,
                                       marker_name_dict, marker_name_config):
    character_list = list(set(list(map(lambda x: x[:x.find(":")], marker_names))))
    character_marker_config_list = []
    character_marker_pos_list = []
    character_marker_vis_list = []
    character_marker_index_list = []
    for character in character_list:
        corresponding_markers = list(filter(lambda x: x[:x.find(":")] == character, marker_names))
        marker_name = list(map(lambda x: x[len(character) + 1:], corresponding_markers))
        marker_name = list(filter(lambda x: x.find("LCM") == -1, marker_name))
        if frozenset(marker_name) in marker_name_dict:
            marker_config = marker_name_dict[frozenset(marker_name)]
        else:
            raise ImportError("Invalid Marker Config")

        character_marker_config_list.append(marker_config)
        marker_list = marker_name_config["body"][marker_config["body"]] + \
                      marker_name_config["finger"][marker_config["finger"]]
        marker_index = [marker_names.index("{}:{}".format(character, marker_name)) for marker_name in marker_list]
        character_marker_pos = marker_pos[:, marker_index, :]
        character_marker_vis = marker_vis[:, marker_index]

        character_marker_index_list.append(marker_index)
        character_marker_pos_list.append(character_marker_pos)
        character_marker_vis_list.append(character_marker_vis)
    return character_list, character_marker_config_list, \
           character_marker_pos_list, character_marker_vis_list, character_marker_index_list


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
                ref_frame_list[0, ii] = back
                break
            back -= 1

        forward = frame + 1
        while forward != anim_length:
            if not marker_vis_arr[forward, invis_marker_idx]:
                ref_frame_list[1, ii] = forward
                break
            forward += 1
        if back == -1 and forward == anim_length:
            raise Exception("Marker {} invisible all the time!")
    return invis_marker_list, ref_frame_list


def fix_marker_pos_per_frame(frame, marker_pos_arr, marker_vis_arr, invis_marker_list, ref_frame_list,
                             dist_matrix_list, dist_matrix_std, ref_marker_num=5):
    marker_num = marker_pos_arr.shape[1]
    fixed_marker_pos = copy.deepcopy(marker_pos_arr[frame])
    anim_length = marker_pos_arr.shape[0]
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

        # visible_marker_list = np.setdiff1d(np.arange(marker_num),
        #                                    np.hstack([np.where(marker_vis_arr[ref_frame])[0], invis_marker_list]))
        # dist_matrix = dist_matrix_list[ref_frame]
        marker_dist_std = dist_matrix_std[:, invis_marker_idx]
        visible_marker_dist_std_with_idx = np.vstack((np.arange(marker_num).reshape(1, -1),
                                                      marker_dist_std.reshape((1, -1))))[:, visible_marker_list]
        ref_marker_idx = visible_marker_dist_std_with_idx[
            0, np.lexsort(visible_marker_dist_std_with_idx)[:ref_marker_num]].astype(int)
        extended_marker_idx = np.hstack([ref_marker_idx, [invis_marker_idx]])
        ref_marker_pos = marker_pos[ref_marker_idx]

        sub_dist_matrix = dist_matrix[extended_marker_idx][:, extended_marker_idx]
        invis_marker_pos = procrustes_zhou(ref_marker_num, ref_marker_pos.transpose(), sub_dist_matrix ** 2)
        fixed_marker_pos[invis_marker_idx] = invis_marker_pos[:, 0]
    return fixed_marker_pos


def fix_marker_pos(character_marker_pos, character_marker_vis):
    dist_matrix_list = calc_dist_matrix(character_marker_pos)
    dist_matrix_std = np.nanstd(dist_matrix_list, axis=0)

    anim_length = character_marker_pos.shape[0]
    fixed_character_marker_pos = copy.deepcopy(character_marker_pos)
    fix_marker_num = np.where(character_marker_vis)[0].shape[0]
    start_time = time.time()
    for frame in range(anim_length):
        if np.any(character_marker_vis[frame]):
            invis_marker_list, ref_frame_list = find_ref_frame(frame, character_marker_vis)
            # print(frame, invis_marker_list, ref_frame_list)
            fixed_marker_pos = fix_marker_pos_per_frame(frame, character_marker_pos, character_marker_vis,
                                                        invis_marker_list, ref_frame_list,
                                                        dist_matrix_list, dist_matrix_std)
            fixed_character_marker_pos[frame] = fixed_marker_pos

    end_time = time.time()
    print("cost {:.4f}s, fixed {} markers".format(end_time - start_time, fix_marker_num))

    return fixed_character_marker_pos


CLEAN_DATASET_CONFIG_PATH = "/home/panxiaoyu/MoCap/Data/Tencent/clean_dataset_config.json"
DATASET_PATH = "/home/panxiaoyu/MoCap/Data/Tencent"
MARKER_NAME_CONFIG_PATH = "configs/marker_names.json"

# generate marker config dict to check the given markers belongs to which marker config
marker_name_config = json.load(open(MARKER_NAME_CONFIG_PATH, "r"))
marker_name_dict = dict()
for body_type in marker_name_config["body"].keys():
    for finger_type in marker_name_config["finger"].keys():
        marker_set = frozenset(marker_name_config["body"][body_type]).union(
            frozenset(marker_name_config["finger"][finger_type]))
        marker_name_dict[marker_set] = {"body": body_type, "finger": finger_type}

clean_dataset_config = json.load(open(CLEAN_DATASET_CONFIG_PATH, "r"))

# for item in tqdm(clean_dataset_config, total=len(clean_dataset_config)):
item = clean_dataset_config[10]
item_folder = item["folder"]
item_name = item["name"]

c3d_path = os.path.join(DATASET_PATH, item_folder, "no_fix_c3d", item_name + ".c3d")

c3d_content = c3d(c3d_path)
c3d_data = c3d_content["data"]
c3d_parameters = c3d_content["parameters"]
c3d_trans = c3d_content["data"]["points"]
c3d_vis = c3d_data["meta_points"]["residuals"]
marker_names = c3d_content['parameters']['POINT']['LABELS']["value"]
marker_vis = c3d_vis[0].transpose()

marker_pos = transform_c3d_marker_pos(c3d_trans)
# recovered_c3d_trans = inverse_transform_c3d_marker_pos(marker_pos)

character_list, character_marker_config_list, character_marker_pos_list, \
character_marker_vis_list, character_marker_index_list = \
    get_character_marker_pos_vis_index(marker_names, marker_pos, marker_vis,
                                       marker_name_dict, marker_name_config)

for character_idx, character_name in enumerate(character_list):
    print("Processing character \"{}\"".format(character_name))
    character_config = character_marker_config_list[character_idx]
    character_marker_pos = character_marker_pos_list[character_idx]
    character_marker_vis = character_marker_vis_list[character_idx]

    fixed_character_marker_pos = fix_marker_pos(character_marker_pos, character_marker_vis)
    fixed_c3d_trans = inverse_transform_c3d_marker_pos(fixed_character_marker_pos)



    c3d_content["data"]["points"] = fixed_c3d_trans
    c3d_data["meta_points"]["residuals"] = np.zeros_like(c3d_data["meta_points"]["residuals"])
    c3d_content.write("fixed.c3d")
# np.savez("fixed.npz", trans=fixed_character_marker_pos, vis=character_marker_vis)

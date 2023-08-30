import numpy as np
import pandas as pd
from .pymo.parsers import BVHParser
from .pymo.writers import BVHWriter
from .pymo.data import MocapData
from scipy.spatial.transform import Rotation
from .motion_utils import *


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


if __name__ == "__main__":
    NPZ_PATH = "/home/panxiaoyu/MoCap/Data/Tencent/part1/10Fingers/20210805_mocap/fbx_npz/01___0.npz"

    npz_content = np.load(NPZ_PATH, allow_pickle=True)

    t_pose_joint_pos = npz_content["t_pose_joint_pos"]
    joint_names = [name.decode() for name in npz_content["name"]]
    joint_orient = npz_content["joint_orient"]
    joint_rot_order = npz_content["joint_rot_order"]
    parent = npz_content["parent"]
    poses = npz_content["poses"]
    trans = npz_content["trans"]
    anim_length = poses.shape[0]

    joint_names_no_prefix = [name[name.find(":") + 1:] for name in joint_names]
    rot_mat_arr = calc_rot_mat(poses, joint_orient, joint_rot_order)

    export_to_bvh("../out.bvh", joint_names_no_prefix, rot_mat_arr, trans, t_pose_joint_pos, parent)

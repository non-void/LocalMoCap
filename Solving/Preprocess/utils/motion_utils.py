import numpy as np
import copy
from scipy.spatial.transform import Rotation


def calc_rot_mat(joint_rotation_euler, joint_orientation_euler, joint_rotation_order):
    anim_length = joint_rotation_euler.shape[0]
    joint_num = joint_rotation_euler.shape[1]
    out_mat = np.zeros((anim_length, joint_num, 3, 3))
    joint_rotation_order = [order.decode("utf-8") for order in joint_rotation_order]

    for joint_idx in range(joint_num):
        x_mat = Rotation.from_euler("xyz", np.hstack([joint_rotation_euler[:, joint_idx, 0].reshape((-1, 1)),
                                                      np.zeros((anim_length, 1)),
                                                      np.zeros((anim_length, 1))]),
                                    degrees=True).as_matrix()
        y_mat = Rotation.from_euler("xyz", np.hstack([np.zeros((anim_length, 1)),
                                                      joint_rotation_euler[:, joint_idx, 1].reshape((-1, 1)),
                                                      np.zeros((anim_length, 1))]),
                                    degrees=True).as_matrix()
        z_mat = Rotation.from_euler("xyz", np.hstack([np.zeros((anim_length, 1)),
                                                      np.zeros((anim_length, 1)),
                                                      joint_rotation_euler[:, joint_idx, 2].reshape((-1, 1)), ]),
                                    degrees=True).as_matrix()
        x_orient_mat = Rotation.from_euler("xyz", [joint_orientation_euler[joint_idx, 0], 0, 0],
                                           degrees=True).as_matrix()
        y_orient_mat = Rotation.from_euler("xyz", [0, joint_orientation_euler[joint_idx, 1], 0],
                                           degrees=True).as_matrix()
        z_orient_mat = Rotation.from_euler("xyz", [0, 0, joint_orientation_euler[joint_idx, 2]],
                                           degrees=True).as_matrix()
        if joint_rotation_order[joint_idx] == "xyz":
            angle_rot_mat = z_mat @ y_mat @ x_mat
        elif joint_rotation_order[joint_idx] == "xzy":
            angle_rot_mat = y_mat @ z_mat @ x_mat
        elif joint_rotation_order[joint_idx] == "yxz":
            angle_rot_mat = z_mat @ x_mat @ y_mat
        elif joint_rotation_order[joint_idx] == "yzx":
            angle_rot_mat = x_mat @ z_mat @ y_mat
        elif joint_rotation_order[joint_idx] == "zxy":
            angle_rot_mat = y_mat @ x_mat @ z_mat
        elif joint_rotation_order[joint_idx] == "zyx":
            angle_rot_mat = x_mat @ y_mat @ z_mat
        joint_orient_mat = z_orient_mat @ y_orient_mat @ x_orient_mat

        rot_mat = joint_orient_mat @ angle_rot_mat
        out_mat[:, joint_idx] = rot_mat
    return out_mat


def batch_get_joint_transform_matrix(rot_mat, t_pose_joint_pos, joint_hierarchy,get_joint_transform=False):
    def with_zeros(x):
        anim_length = x.shape[0]
        return np.concatenate([x, np.tile(np.array([[[0, 0, 0, 1]]]), (anim_length, 1, 1))], axis=1)

    def pack(x):
        anim_length, joint_num = x.shape[0], x.shape[1]
        return np.concatenate([np.zeros((anim_length, joint_num, 4, 3)), x], axis=3)

    anim_length, joint_num = rot_mat.shape[0], rot_mat.shape[1]
    R = rot_mat

    G = np.empty((anim_length, joint_num, 4, 4))
    G[:, 0] = with_zeros(np.concatenate([R[:, 0],
                                         np.tile(t_pose_joint_pos[0].reshape((1, 3, 1)), (anim_length, 1, 1))], axis=2))
    for i in range(1, len(joint_hierarchy)):
        a = G[:, joint_hierarchy[i]]
        b = with_zeros(np.concatenate([R[:, i], np.tile(
            (t_pose_joint_pos[i] - t_pose_joint_pos[joint_hierarchy[i]]).reshape((1, 3, 1)),
            (anim_length, 1, 1))], axis=2))
        G[:, i] = np.matmul(a, b)
    global_joint_pos = copy.deepcopy(G[:, :, :3, 3])

    if not get_joint_transform:
        GG = pack(np.matmul(G, np.concatenate([np.tile(t_pose_joint_pos.reshape((1, -1, 3)), (anim_length, 1, 1)),
                                               np.zeros((anim_length, joint_num, 1))], axis=2). \
                            reshape((anim_length, joint_num, 4, 1))))
        G -= GG
    return global_joint_pos, G

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


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

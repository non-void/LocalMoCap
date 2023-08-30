import numpy as np
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

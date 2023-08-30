import numpy as np
import copy

def transform_c3d_marker_pos(c3d_trans):
    trans_mat = np.array([[-0.1, 0, 0, 0],
                          [0, 0, 0.1, 0],
                          [0, 0.1, 0, 0],
                          [0, 0, 0, 1]])
    transformed_c3d_trans = np.einsum("ab,bcd->acd", trans_mat, c3d_trans)
    marker_pos = transformed_c3d_trans.transpose([2, 1, 0])[:, :, :3]
    return marker_pos


def inverse_transform_c3d_marker_pos(marker_pos):
    trans_mat = np.array([[-10.0, 0, 0],
                          [0, 0, 10.0],
                          [0, 10.0, 0]])
    transformed_marker_pos = np.einsum("ab,bcd->acd", trans_mat, marker_pos.transpose())
    res = np.zeros((4, transformed_marker_pos.shape[1], transformed_marker_pos.shape[2]))
    res[:3] = transformed_marker_pos
    res[3] = 1
    return res

def remove_marker_global_transform(marker_pos, global_rotation_mat=None, global_trans=None):
    marker_pos_h = np.concatenate([copy.deepcopy(marker_pos), np.ones((marker_pos.shape[0], marker_pos.shape[1], 1))],
                                  axis=2)
    eye_mat_arr = np.repeat(np.eye(4).reshape((1, 4, 4)), marker_pos.shape[0], axis=0)
    trans_mat_arr = copy.deepcopy(eye_mat_arr)
    rot_mat_arr = copy.deepcopy(eye_mat_arr)

    if global_rotation_mat is not None:
        rot_mat_arr[:, :3, :3] = np.linalg.inv(global_rotation_mat)
    if global_trans is not None:
        trans_mat_arr[:, :3, 3] = -global_trans

    transform_mat_arr = rot_mat_arr @ trans_mat_arr
    out = np.swapaxes((transform_mat_arr @ np.swapaxes(marker_pos_h, 1, 2))[:, :3, :], 1, 2)

    # transformed_marker_pos=copy.deepcopy(marker_pos)
    # if global_trans is not None:
    #     transformed_marker_pos = transformed_marker_pos - \
    #                              global_trans.reshape((global_trans.shape[0], 1, global_trans.shape[1]))
    # if global_rotation_mat is not None:
    #     transformed_marker_pos = np.einsum("abc,adc->adb", np.linalg.inv(global_rotation_mat), transformed_marker_pos)
    return out


def add_marker_global_transform(marker_pos, global_rotation_mat=None, global_trans=None):
    marker_pos_h = np.concatenate([copy.deepcopy(marker_pos), np.ones((marker_pos.shape[0], marker_pos.shape[1], 1))],
                                  axis=2)
    eye_mat_arr = np.repeat(np.eye(4).reshape((1, 4, 4)), marker_pos.shape[0], axis=0)
    trans_mat_arr = copy.deepcopy(eye_mat_arr)
    rot_mat_arr = copy.deepcopy(eye_mat_arr)

    if global_rotation_mat is not None:
        rot_mat_arr[:, :3, :3] = global_rotation_mat
    if global_trans is not None:
        trans_mat_arr[:, :3, 3] = global_trans

    transform_mat_arr = trans_mat_arr @ rot_mat_arr
    out = np.swapaxes((transform_mat_arr @ np.swapaxes(marker_pos_h, 1, 2))[:, :3, :], 1, 2)
    return out

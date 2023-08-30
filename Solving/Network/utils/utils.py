import numpy as np
import json
import os
import random
import h5py
import torch
import torch.nn as nn

BODY_JOINT_NAME_LIST_PATH = "../Assets/joint_config_body.txt"
LEFTHAND_JOINT_CONFIG_PATH = "../Assets/joint_config_left_hand.txt"
RIGHTHAND_JOINT_CONFIG_PATH = "../Assets/joint_config_right_hand.txt"
MARKER_NAME_CONFIG_PATH = "../Assets/marker_names.json"
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

marker_name_config = json.load(open(MARKER_NAME_CONFIG_PATH, "r"))
PRODUCTION_MARKER_NAME_LIST = marker_name_config["body"]["production"]
FRONT_WAIST_MARKER_NAME_LIST = marker_name_config["body"]["front_waist"]
THREE_FINGER_MARKER_NAME_LIST = marker_name_config["finger"]["3"]
FIVE_FINGER_MARKER_NAME_LIST = marker_name_config["finger"]["5"]
TEN_FINGER_MARKER_NAME_LIST = marker_name_config["finger"]["10"]

REAL_HEAD_JOINT_NAME_LIST = ["Neck1", "Head", "HeadEnd"]
REAL_SHOULDER_JOINT_NAME_LIST = ["LeftShoulder", "RightShoulder",
                                 "LeftArm", "RightArm"]
REAL_ARM_JOINT_NAME_LIST = ["LeftForeArm", "RightForeArm"]
REAL_WRIST_JOINT_NAME_LIST = ["LeftHand", "RightHand"]
REAL_TORSO_JOINT_NAME_LIST = ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck",
                              "LeftUpLeg", "RightUpLeg"]
REAL_THIGH_JOINT_NAME_LIST = ["LeftLeg", "RightLeg"]
REAL_FOOT_JOINT_NAME_LIST = ["LeftFoot", "RightFoot",
                             "LeftToeBase", "RightToeBase",
                             "LeftToeBaseEnd", "RightToeBaseEnd"]

PRODUCTION_HEAD_MARKER_NAME_LIST = ["ARIEL",
                                    "LBHD", "LFHD",
                                    "RBHD", "RFHD"]
PRODUCTION_SHOULDER_MARKER_NAME_LIST = ["LTSH", "LBSH", "LFSH",
                                        "RTSH", "RBSH", "RFSH"]
PRODUCTION_ARM_MARKER_NAME_LIST = ["LUPA", "LELB", "LBEL",
                                   "RUPA", "RELB", "RBEL"]
PRODUCTION_WRIST_MARKER_NAME_LIST = ["LIHAND", "LOHAND", "LIWR", "LOWR", "LFRM",
                                     "RIHAND", "ROHAND", "RIWR", "ROWR", "RFRM"]
PRODUCTION_TORSO_MARKER_NAME_LIST = ["CLAV", "STRN", "C7", "T10", "LT10", "RT10",
                                     "LFWT", "LMWT", "LBWT",
                                     "RFWT", "RMWT", "RBWT"]
PRODUCTION_THIGH_MARKER_NAME_LIST = ["LKNU", "LKNE", "LSHN", "LTHI",
                                     "RKNU", "RKNE", "RSHN", "RTHI"]
PRODUCTION_FOOT_MARKER_NAME_LIST = ["LANK", "LMT1", "LTOE", "LMT5", "LHEL",
                                    "RANK", "RMT1", "RTOE", "RMT5", "RHEL"]

SYNTHETIC_HEAD_JOINT_NAME_LIST = ["Head"]
SYNTHETIC_SHOULDER_JOINT_NAME_LIST = ['L_Collar', 'R_Collar', 'L_Shoulder', 'R_Shoulder']
SYNTHETIC_ARM_JOINT_NAME_LIST = ['L_Elbow', 'R_Elbow']
SYNTHETIC_WRIST_JOINT_NAME_LIST = ['L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
SYNTHETIC_TORSO_JOINT_NAME_LIST = ['Pelvis', 'Spine1', 'Spine2', 'Spine3', 'L_Hip', 'R_Hip', 'Neck']
SYNTHETIC_THIGH_JOINT_NAME_LIST = ['L_Knee', 'R_Knee']
SYNTHETIC_FOOT_JOINT_NAME_LIST = ['L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot']

SYNTHETIC_HEAD_MARKER_NAME_LIST = ["ARIEL",
                                   "LBHD", "LFHD",
                                   "RBHD", "RFHD"]
SYNTHETIC_SHOULDER_MARKER_NAME_LIST = ["LTSH", "LBSH", "LFSH",
                                       "RTSH", "RBSH", "RFSH"]
SYNTHETIC_ARM_MARKER_NAME_LIST = ["LUPA", "LIEL", "LELB", "LWRE",
                                  "RUPA", "RIEL", "RELB", "RWRE"]
SYNTHETIC_WRIST_MARKER_NAME_LIST = ["LOWR", "LIWR", "LIHAND", "LOHAND",
                                    "ROWR", "RIWR", "RIHAND", "ROHAND"]
SYNTHETIC_TORSO_MARKER_NAME_LIST = ["CLAV", "STRN", "C7", "T10", "L4",
                                    "LFWT", "LMWT", "LBWT",
                                    "RFWT", "RMWT", "RBWT"]
SYNTHETIC_THIGH_MARKER_NAME_LIST = ["LKNI", "LKNE", "LHIP", "LSHN",
                                    "RKNI", "RKNE", "RHIP", "RSHN"]
SYNTHETIC_FOOT_MARKER_NAME_LIST = ["LANK", "LHEL", "LMT1", "LTOE", "LMT5",
                                   "RANK", "RHEL", "RMT1", "RTOE", "RMT5"]

REAL_HEAD_JOINT_LOSS_FACTOR = 10
REAL_SHOULDER_JOINT_LOSS_FACTOR = 5
REAL_ARM_JOINT_LOSS_FACTOR = 8
REAL_WRIST_JOINT_LOSS_FACTOR = 10
REAL_TORSO_JOINT_LOSS_FACTOR = 5
REAL_THIGH_JOINT_LOSS_FACTOR = 8
REAL_FOOT_JOINT_LOSS_FACTOR = 10

PRODUCTION_HEAD_MARKER_LOSS_FACTOR = 10
PRODUCTION_SHOULDER_MARKER_LOSS_FACTOR = 5
PRODUCTION_ARM_MARKER_LOSS_FACTOR = 8
PRODUCTION_WRIST_MARKER_LOSS_FACTOR = 10
PRODUCTION_TORSO_MARKER_LOSS_FACTOR = 5
PRODUCTION_THIGH_MARKER_LOSS_FACTOR = 8
PRODUCTION_FOOT_MARKER_LOSS_FACTOR = 10

real_body_joint_loss_weight = np.ones(len(BODY_JOINT_NAME_LIST))

real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_HEAD_JOINT_NAME_LIST]] *= REAL_HEAD_JOINT_LOSS_FACTOR
real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_SHOULDER_JOINT_NAME_LIST]] *= REAL_SHOULDER_JOINT_LOSS_FACTOR
real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_ARM_JOINT_NAME_LIST]] *= REAL_ARM_JOINT_LOSS_FACTOR
real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_WRIST_JOINT_NAME_LIST]] *= REAL_WRIST_JOINT_LOSS_FACTOR
real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_TORSO_JOINT_NAME_LIST]] *= REAL_TORSO_JOINT_LOSS_FACTOR
real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_THIGH_JOINT_NAME_LIST]] *= REAL_THIGH_JOINT_LOSS_FACTOR
real_body_joint_loss_weight[[BODY_JOINT_NAME_LIST.index(joint_name) \
                             for joint_name in REAL_FOOT_JOINT_NAME_LIST]] *= REAL_FOOT_JOINT_LOSS_FACTOR

production_marker_loss_weight = np.ones(len(PRODUCTION_MARKER_NAME_LIST))

production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_HEAD_MARKER_NAME_LIST]] *= PRODUCTION_HEAD_MARKER_LOSS_FACTOR
production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_SHOULDER_MARKER_NAME_LIST]] *= PRODUCTION_SHOULDER_MARKER_LOSS_FACTOR
production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_ARM_MARKER_NAME_LIST]] *= PRODUCTION_ARM_MARKER_LOSS_FACTOR
production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_WRIST_MARKER_NAME_LIST]] *= PRODUCTION_WRIST_MARKER_LOSS_FACTOR
production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_TORSO_MARKER_NAME_LIST]] *= PRODUCTION_TORSO_MARKER_LOSS_FACTOR
production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_THIGH_MARKER_NAME_LIST]] *= PRODUCTION_THIGH_MARKER_LOSS_FACTOR
production_marker_loss_weight[[PRODUCTION_MARKER_NAME_LIST.index(marker_name) for marker_name in
                               PRODUCTION_FOOT_MARKER_NAME_LIST]] *= PRODUCTION_FOOT_MARKER_LOSS_FACTOR


class Criterion_EE:
    def __init__(self, args, base_criterion, norm_eps=0.008):
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps

    def __call__(self, pred, gt):
        reg_ee_loss = self.base_criterion(pred, gt)
        if self.args.ee_velo:
            gt_norm = torch.norm(gt, dim=-1)
            contact_idx = gt_norm < self.norm_eps
            extra_ee_loss = self.base_criterion(pred[contact_idx], gt[contact_idx])
        else:
            extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return []


class Eval_Criterion:
    def __init__(self, parent):
        self.pa = parent
        self.base_criterion = nn.MSELoss()
        pass

    def __call__(self, pred, gt):
        for i in range(1, len(self.pa)):
            pred[..., i, :] += pred[..., self.pa[i], :]
            gt[..., i, :] += pred[..., self.pa[i], :]
        return self.base_criterion(pred, gt)


class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.HUBER_DELTA = delta

    def forward(self, input, target):
        error_mat = input - target
        _error_ = torch.sqrt(torch.sum(error_mat ** 2))
        HUBER_DELTA = self.HUBER_DELTA
        switch_l = _error_ < HUBER_DELTA
        switch_2 = _error_ >= HUBER_DELTA
        x = switch_l * (0.5 * _error_ ** 2) + switch_2 * (
                0.5 * HUBER_DELTA ** 2 + HUBER_DELTA * (_error_ - HUBER_DELTA))
        return x


class angle_loss(nn.Module):
    def __init__(self):
        super(angle_loss, self).__init__()

    def forward(self, input, target):
        y_pred1 = input[..., 0, :]
        y_pred2 = input[..., 1, :]
        y_pred3 = input[..., 2, :]
        y_pred1 = y_pred1.view(-1, 1, 3)
        y_pred2 = y_pred2.view(-1, 1, 3)
        y_pred3 = y_pred3.view(-1, 1, 3)
        y_pred1 = y_pred1.repeat([1, 3, 1])
        y_pred2 = y_pred2.repeat([1, 3, 1])
        y_pred3 = y_pred3.repeat([1, 3, 1])
        target = target.view(-1, 3, 3)
        z_pred1 = target * y_pred1
        z_pred2 = target * y_pred2
        z_pred3 = target * y_pred3
        z_pred1 = torch.sum(z_pred1, axis=-1)
        z_pred2 = torch.sum(z_pred2, axis=-1)
        z_pred3 = torch.sum(z_pred3, axis=-1)
        z_pred1 = z_pred1.view(-1, 3, 1)
        z_pred2 = z_pred2.view(-1, 3, 1)
        z_pred3 = z_pred3.view(-1, 3, 1)
        z_pred = torch.cat([z_pred1, z_pred2, z_pred3], axis=2)
        # z_pred_trace = torch.trace(z_pred)
        z_pred_trace = z_pred[:, 0, 0] + z_pred[:, 1, 1] + z_pred[:, 2, 2]
        z_pred_trace = (z_pred_trace - 1.) / 2.0000000000
        z_pred_trace = torch.clamp(z_pred_trace, -1.0, 1.0)
        z_pred_trace = torch.acos(z_pred_trace)
        z_pred_trace = z_pred_trace * 180. / 3.141592653
        error = torch.mean(z_pred_trace)
        return error


class quat_angle_loss(nn.Module):
    def __init__(self):
        super(quat_angle_loss, self).__init__()

    def nanmean(self, v, *args, inplace=False, **kwargs):
        if not inplace:
            v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    def forward(self, target, pred):
        normed_pred_rot = pred / torch.unsqueeze(torch.linalg.norm(pred, dim=3), dim=3)
        angle_error = torch.arccos(torch.abs(torch.sum((normed_pred_rot * target), dim=3))) * 2 * (180 / np.pi)
        angle_error = self.nanmean(angle_error)
        return angle_error


class vertex_loss(nn.Module):
    def __init__(self):
        super(vertex_loss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.sqrt(torch.sum(torch.pow(input - target, 2), dim=-1)))


def show_gpu_tensor(x):
    return x.detach().cpu().numpy()


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def rotmat2angle(mat):
    diag = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
    angle = torch.arccos(torch.clip((diag - 1) / 2, -1, 1))
    angle = torch.abs(angle) * 180 / 3.141592653
    return angle


def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    qx = quat[..., 0]
    qy = quat[..., 1]
    qz = quat[..., 2]
    qw = quat[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quat.shape[:-1] + (3, 3), device=quat.device)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def euler2rotmat(euler):
    c1 = torch.cos(euler[..., 0])
    s1 = torch.sin(euler[..., 0])
    c2 = torch.cos(euler[..., 1])
    s2 = torch.sin(euler[..., 1])
    c3 = torch.cos(euler[..., 2])
    s3 = torch.sin(euler[..., 2])

    m = torch.empty(euler.shape[:-1] + (3, 3), device=euler.device)

    m[..., 0, 0] = c2 * c3
    m[..., 0, 1] = c3 * s2 * s1 - s3 * c1
    m[..., 0, 2] = c3 * s2 * c1 + s3 * s1
    m[..., 1, 0] = s3 * c2
    m[..., 1, 1] = c3 * c1 + s3 * s2 * s1
    m[..., 1, 2] = s3 * s2 * c1 - s1 * c3
    m[..., 2, 0] = -s2
    m[..., 2, 1] = c2 * s1
    m[..., 2, 2] = c2 * c1

    return m


def quat2euler(quater: torch.Tensor) -> torch.Tensor:
    x = quater[..., 0]
    y = quater[..., 1]
    z = quater[..., 2]
    w = quater[..., 3]

    m = torch.empty(quater.shape[:-1] + (3,), device=quater.device)

    m[..., 0] = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    m[..., 1] = torch.arcsin(2 * (w * y - x * z))
    m[..., 2] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

    return m


def set_seed(seed: int):
    # ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

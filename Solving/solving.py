import os
import argparse
import torch
from tqdm import tqdm
import warnings

from Network.marker_to_pose import marker_to_pose


def main(args, in_file_path, out_file_path):
    if args.have_warning:
        warnings.filterwarnings("ignore")

    marker_to_pose(args, in_file_path, out_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--have_warning", type=bool, default=True)

    parser.add_argument("--act_fun", type=str, nargs="+",
                        default=["Mish", "ReLU", "ReLU", "ReLU"])
    parser.add_argument("--network_dim", type=int, nargs='+',
                        default=[0, 1024, 1024, 1024, 0])
    parser.add_argument("--marker_gcn_dim", type=int, nargs="+",
                        default=[4, 8, 16])
    parser.add_argument("--marker_gcn_act", type=str, nargs="+",
                        default=["Mish", "Mish", "Mish"])

    parser.add_argument("--skeleton_gcn_dim", type=int, nargs="+",
                        default=[40, 20, 0])
    parser.add_argument("--skeleton_gcn_act", type=str, nargs="+",
                        default=["Mish", "Mish", "Mish"])

    parser.add_argument("--marker_skeleton_conv_gcn_dim", type=int, nargs="+",
                        default=[[16, 30], [30, 40], [40, 30]])
    parser.add_argument("--marker_skeleton_conv_gcn_act", type=str, nargs="+",
                        default=["Mish", "Mish", "Mish"])
    parser.add_argument("--use_cuda", type=bool, default=True, help="enable cuda")

    parser.add_argument("--root_dir", type=str,
                        default=".")
    parser.add_argument("--marker_name_config_path", type=str,
                        default="configs/marker_names.json")
    parser.add_argument("--full_joint_name_config_path", type=str,
                        default="configs/synthetic_joint_config_full.txt")
    parser.add_argument("--body_joint_name_config_path", type=str,
                        default="configs/synthetic_joint_config_body.txt")
    parser.add_argument("--left_hand_joint_name_config_path", type=str,
                        default="configs/synthetic_joint_config_left_hand.txt")
    parser.add_argument("--right_hand_joint_name_config_path", type=str,
                        default="configs/synthetic_joint_config_right_hand.txt")

    parser.add_argument("--ten_finger_left_hand_resume_file_path", type=str,
                        default="Network/checkpoints/solving/left_hand.pth.tar")
    parser.add_argument("--ten_finger_left_hand_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/fix_occlusion_marker_pos.npz")
    parser.add_argument("--ten_finger_left_wrist_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/wrist_marker_pos.npz")
    parser.add_argument("--ten_finger_left_hand_motion_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/motion_mat.npz")
    parser.add_argument("--ten_finger_left_hand_weight_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/weights.npz")
    parser.add_argument("--ten_finger_left_hand_joint_offset_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/t_pose.npz")

    parser.add_argument("--ten_finger_right_hand_resume_file_path", type=str,
                        default="Network/checkpoints/solving/right_hand.pth.tar")
    parser.add_argument("--ten_finger_right_hand_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/fix_occlusion_marker_pos.npz")
    parser.add_argument("--ten_finger_right_wrist_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/wrist_marker_pos.npz")
    parser.add_argument("--ten_finger_right_hand_motion_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/motion_mat.npz")
    parser.add_argument("--ten_finger_right_hand_weight_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/weights.npz")
    parser.add_argument("--ten_finger_right_hand_joint_offset_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/t_pose.npz")
    parser.add_argument("--hand_connect_mat_path", type=str,
                        default="Network/Assets/connect_mat/10_finger_mat_synthetic.npy")

    parser.add_argument("--front_waist_body_resume_file_path", type=str,
                        default="Network/checkpoints/solving/body.pth.tar")
    parser.add_argument("--front_waist_body_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_front_waist/body/fix_occlusion_marker_pos.npz")
    parser.add_argument("--front_waist_body_motion_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_front_waist/body/motion_mat.npz")
    parser.add_argument("--front_waist_body_weight_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_front_waist/body/weights.npz")
    parser.add_argument("--front_waist_body_joint_offset_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_front_waist/body/t_pose.npz")
    parser.add_argument("--body_connect_mat_path", type=str,
                        default="Network/Assets/connect_mat/synthetic_mat_std_3.npy")

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    # sample
    IN_FILE_PATH = "intermediate/network_32_01_poses.npzs10_spheresmall_pass_1_stageii.npz"
    OUT_FILE_PATH = "SampleData/network_32_01_poses.npzs10_spheresmall_pass_1_stageii.bvh"
    main(args, IN_FILE_PATH, OUT_FILE_PATH)

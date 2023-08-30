import os
import argparse
import torch
from tqdm import tqdm
import warnings

from Preprocess.fix_occlusion_jump_synthetic import fix_data_traditional_method
from Network.network_fix_occlusion import network_fix_occlusion


def main(args, in_file_path, out_file_path,
         use_traditional=True, use_network_fix_occlusion=True):
    if args.have_warning:
        warnings.filterwarnings("ignore")

    in_file_base_name = os.path.basename(in_file_path)
    intermediate_file_path = "intermediate/{}".format(in_file_base_name)
    network_fixed_file_path = "intermediate/network_{}".format(in_file_base_name)

    if use_traditional:
        print("Using traditional methods to fix occlusions")
        fix_data_traditional_method(args, in_file_path, intermediate_file_path, fix_marker_jump=True)
    if use_network_fix_occlusion:
        print()
        print("Using network to fix occlusions")
        network_fix_occlusion(args, in_file_path, intermediate_file_path, network_fixed_file_path)

        print()
        print("Fixing marker jumps")
        fix_data_traditional_method(args, network_fixed_file_path, out_file_path,
                                    fix_occlusion=False, fix_marker_jump=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--have_warning", type=bool, default=True)

    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--network_type", type=str, default="BiLSTM")
    parser.add_argument("--body_hidden_dim", type=int, default=800)
    parser.add_argument("--hand_hidden_dim", type=int, default=400)
    parser.add_argument("--lstm_layer_num", type=int, default=1)

    parser.add_argument("--marker_gcn_dim", type=int, nargs="+",
                        default=[6, 10, 16])
    parser.add_argument("--marker_gcn_act", type=str, nargs="+",
                        default=["Mish", "Mish", "Mish"])

    parser.add_argument("--root_dir", type=str,
                        default=".")
    parser.add_argument("--marker_name_config_path", type=str,
                        default="configs/marker_names.json")
    parser.add_argument("--body_joint_name_config_path", type=str,
                        default="configs/joint_config_body.txt")
    parser.add_argument("--left_hand_joint_name_config_path", type=str,
                        default="configs/joint_config_left_hand.txt")
    parser.add_argument("--right_hand_joint_name_config_path", type=str,
                        default="configs/joint_config_right_hand.txt")

    parser.add_argument("--fix_occlusion_ref_marker_method", type=str, default="mean")
    parser.add_argument("--fix_occlusion_ref_marker_num", type=int, default=6)

    parser.add_argument("--fix_waist_jump_peak_threshold", type=float, default=0.5)
    parser.add_argument("--fix_waist_jump_peak_interval", type=int, default=2)
    parser.add_argument("--fix_waist_jump_loop_num", type=int, default=2)
    parser.add_argument("--fix_body_jump_peak_threshold", type=float, default=0.5)
    parser.add_argument("--fix_body_jump_peak_interval", type=int, default=2)
    parser.add_argument("--fix_body_jump_loop_num", type=int, default=2)
    parser.add_argument("--fix_hand_jump_peak_threshold", type=float, default=0.2)
    parser.add_argument("--fix_hand_jump_peak_interval", type=int, default=2)
    parser.add_argument("--fix_hand_jump_loop_num", type=int, default=2)

    parser.add_argument("--ten_finger_left_hand_resume_file_path", type=str,
                        default="Network/checkpoints/preprocess/left_hand.pth.tar")
    parser.add_argument("--ten_finger_left_hand_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/fix_occlusion_marker_pos.npz")
    parser.add_argument("--ten_finger_left_wrist_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/left_hand/wrist_marker_pos.npz")

    parser.add_argument("--ten_finger_right_hand_resume_file_path", type=str,
                        default="Network/checkpoints/preprocess/right_hand.pth.tar")
    parser.add_argument("--ten_finger_right_hand_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/fix_occlusion_marker_pos.npz")
    parser.add_argument("--ten_finger_right_wrist_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_10fingers/right_hand/wrist_marker_pos.npz")
    parser.add_argument("--hand_connect_mat_path", type=str,
                        default="Network/Assets/connect_mat/10_finger_mat_synthetic.npy")

    parser.add_argument("--front_waist_body_resume_file_path", type=str,
                        default="Network/checkpoints/preprocess/body.pth.tar")
    parser.add_argument("--front_waist_body_marker_pos_mean_std_path", type=str,
                        default="Network/Assets/mean_std/synthetic_front_waist/body/fix_occlusion_marker_pos.npz")
    parser.add_argument("--body_connect_mat_path", type=str,
                        default="Network/Assets/connect_mat/synthetic_mat_std_3.npy")

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    # sample
    IN_FILE_PATH = "SampleData/32_01_poses.npzs10_spheresmall_pass_1_stageii.npz"
    OUT_FILE_PATH = "SampleData/32_01_poses.npzs10_spheresmall_pass_1_stageii_fixed.npz"
    main(args, IN_FILE_PATH, OUT_FILE_PATH)

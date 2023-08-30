import torch
import os
import smplx
from external.smplx.smplx.lbs import batch_rodrigues
from tqdm import tqdm
import numpy as np
import time
import copy
import argparse
import random
import json

parser = argparse.ArgumentParser()

parser.add_argument('--body_occ_rate', type=float, default=0.2)
parser.add_argument('--finger_occ_rate', type=float, default=0.2)
parser.add_argument('--body_shift_prob', type=float, default=0.0005)
parser.add_argument('--finger_shift_prob', type=float, default=0.0005)
parser.add_argument('--body_shift_scale', type=float, default=0.05)
parser.add_argument('--finger_shift_scale', type=float, default=0.02)
parser.add_argument('--config', type=str, choices=['production_5', 'front_waist_10'])
parser.add_argument('--save_name', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--seed', type=int, default=7777)

opts = parser.parse_args()
preprocess_path = './data'
gap_info_dict = {}
gap_info_dict["body"] = {}
gap_info_dict["finger"] = {}
gap_info_dict["body"]["production"] = np.load(os.path.join(preprocess_path, "gap_info/production.npz"))
gap_info_dict['body']['front_waist'] = np.load(os.path.join(preprocess_path, 'gap_info/front_waist.npz'))
gap_info_dict["finger"]["5"] = np.load(os.path.join(preprocess_path, "gap_info/five_finger.npz"))
gap_info_dict['finger']['10'] = np.load(os.path.join(preprocess_path, 'gap_info/ten_finger.npz'))

def set_seed(seed: int):
    # ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def corrupt_occ(character_config, marker, body_ratio, finger_ratio):
    ret = copy.deepcopy(marker)
    anim_length = marker.shape[0]
    sample_ratio = {
        'body': body_ratio,
        'finger': finger_ratio
    }
    sample_strategy = 'gap'
    gap_length_clip = np.array([1, 200], dtype=int)

    regenerate = True
    regenerate_time = 0
    while regenerate and regenerate_time < 5:
        regenerate_time += 1
        corrupted_marker_vis = sample_vis_arr(character_config, anim_length, sample_ratio, sample_strategy, gap_length_clip)
        missing_ratio = np.sum(corrupted_marker_vis)/(corrupted_marker_vis.shape[0]*corrupted_marker_vis.shape[1])
        if body_ratio - missing_ratio > 0.02:
            regenerate = True
            gap_length_clip = (gap_length_clip//2).astype(int)
            if gap_length_clip[0] == 0:
                gap_length_clip[0] = 1
        else:
            regenerate = False
    for i in range(3):
        ret[:, :, i] = marker[:, :, i] * (1 - corrupted_marker_vis)
    return ret, corrupted_marker_vis


def corrupt_shift(character_config, marker, body_sigma, finger_sigma, body_beta, finger_beta):
    ret = copy.deepcopy(marker)
    u_normal = 0
    size_normal = 1
    if character_config['body'] == 'front_waist':
        body_size_bernoulli = 53
    elif character_config['body'] == 'production':
        body_size_bernoulli = 57
    finger_size_bernoulli = int(character_config['finger']) * 2
    for i in range(marker.shape[0]):
        body_alpha = np.random.normal(loc=u_normal, scale=body_sigma, size=1)
        finger_alpha = np.random.normal(loc=u_normal, scale=finger_sigma, size=1)
        min_body_alpha = min(abs(body_alpha), 2 * body_sigma)
        min_finger_alpha = min(abs(finger_alpha), 2 * finger_sigma)
        x_shift_body = np.random.binomial(n=1, p=min_body_alpha, size=body_size_bernoulli)[:, np.newaxis]
        x_shift_finger = np.random.binomial(n=1, p=min_finger_alpha, size=finger_size_bernoulli)[:, np.newaxis]
        x_v_body = np.random.uniform(-body_beta, body_beta, size=(body_size_bernoulli, 3))
        x_v_finger = np.random.uniform(-finger_beta, finger_beta, size=(finger_size_bernoulli, 3))
        ret[i] = ret[i] + np.concatenate([x_shift_body * x_v_body, x_shift_finger * x_v_finger])
    return ret


def sample_vis_arr(character_config, anim_length, sample_ratio, sample_strategy, gap_length_clip=[1, 200]):
    def generate_vis_arr(anim_length, ratio, marker_missing_rate, gap_frequency, strategy="random"):
        marker_num = marker_missing_rate.shape[0]
        ret = np.zeros((anim_length, marker_num), dtype=int)
        normed_marker_missing_rate = marker_missing_rate / np.mean(marker_missing_rate) * ratio
        normed_marker_missing_rate[normed_marker_missing_rate>0.5]=0.5
        missing_marker_num = anim_length * normed_marker_missing_rate
        start_time = time.time()
        reinit = False
        gap_arr_length = gap_frequency.shape[0]

        if strategy == "random":
            missing_marker_num = np.around(missing_marker_num).astype(int)
            for i in range(marker_num):
                ret[np.random.choice(range(anim_length), missing_marker_num[i], replace=False), i] = 1
        elif strategy == "gap":
            reinit_time=0
            while not reinit and reinit_time<5:
                gap_num_frequency = (gap_frequency * np.arange(gap_arr_length)) / \
                                    np.sum(gap_frequency * np.arange(gap_arr_length))
                gap_num = missing_marker_num.reshape((marker_num, 1)) * gap_num_frequency.reshape((1, -1)) / \
                          np.arange(gap_arr_length)
                gap_num = np.nan_to_num(gap_num)
                gap_possibility, gap_num = np.modf(gap_num)
                random_chosen_gaps = np.random.binomial(1, gap_possibility)
                if not np.any(np.sum((gap_num + random_chosen_gaps) * np.arange(gap_arr_length), axis=1) > anim_length):
                    gap_num += random_chosen_gaps
                gap_num = gap_num.astype(int)
                for marker_idx in range(marker_num):
                    marker_gap_num = gap_num[marker_idx]
                    for gap_length in range(marker_gap_num.shape[0] - 1, 0, -1):
                        for j in range(marker_gap_num[gap_length]):
                            chosen = False
                            while (not chosen) and (not reinit):
                                starting_pos = np.random.choice(anim_length)
                                if starting_pos < anim_length - gap_length - 2 and starting_pos != 0:
                                    if not np.any(ret[starting_pos - 1:starting_pos + gap_length + 1, marker_idx]):
                                        chosen = True
                                        ret[starting_pos:starting_pos + gap_length, marker_idx] = 1
                                if time.time() - start_time > 3:
                                    reinit = True
                                    reinit_time+=1
                if not reinit:
                    break
                else:
                    ret *= 0
                    reinit = False
                    start_time = time.time()
                    print("random gap reinit")
        return ret
    # print(gap_length_clip)
    body_gap_info = gap_info_dict["body"][character_config["body"]]
    body_marker_missing_rate = body_gap_info["missing_arr"][:, 0] / body_gap_info["missing_arr"][:, 1]
    body_marker_gap_frequency = np.sum(body_gap_info["gap_arr"], axis=0) / np.sum(body_gap_info["gap_arr"])

    body_marker_gap_frequency[:gap_length_clip[0]] = 0
    body_marker_gap_frequency[gap_length_clip[1]:] = 0
    body_marker_gap_frequency /= np.sum(body_marker_gap_frequency)

    body_vis_arr = generate_vis_arr(anim_length, sample_ratio["body"], body_marker_missing_rate,
                                    body_marker_gap_frequency, sample_strategy)
    ret = copy.deepcopy(body_vis_arr)
    if character_config["finger"] != "0":
        finger_gap_info = gap_info_dict["finger"][character_config["finger"]]
        finger_marker_missing_rate = finger_gap_info["missing_arr"][:, 0] / finger_gap_info["missing_arr"][:, 1]
        finger_marker_gap_frequency = np.sum(finger_gap_info["gap_arr"], axis=0) / np.sum(finger_gap_info["gap_arr"])

        finger_marker_gap_frequency[:gap_length_clip[0]] = 0
        finger_marker_gap_frequency[gap_length_clip[1]:] = 0
        finger_marker_gap_frequency /= np.sum(finger_marker_gap_frequency)

        finger_vis_arr = generate_vis_arr(anim_length, sample_ratio["finger"],
                                          finger_marker_missing_rate,
                                          finger_marker_gap_frequency,
                                          sample_strategy)
        ret = np.hstack([body_vis_arr, finger_vis_arr])
    return ret


def merge_motion(body_provider, hand_provider):
    body_len = body_provider.shape[0]
    hand_len = hand_provider.shape[0]
    cursor = 0
    while body_len >= hand_len:
        body_provider[cursor:cursor + hand_len, 66:] = hand_provider[:, 66:]
        cursor += hand_len
        body_len -= hand_len
        hand_provider = np.flip(hand_provider, 0)
    body_provider[cursor:, 66:] = hand_provider[:body_len, 66:]
    return body_provider



def generate_data():
    data_list = []
    total_frame = 0
    production_5_vertex_idx = [411, 3, 179, 3513, 3693, 2878, 3017, 891, 4377, 3078, 3077, 1893, 1822, 1854, 1393, 1666, 1658,1602, 2234, 2107, 2134, 2001, 5354, 5342, 6447, 4862, 5123, 5129, 5059, 5659, 5567, 5596, 5462, 3156, 1794, 3101, 6573, 5256, 6525, 850, 1053, 1043, 1097, 3327, 3387, 3347, 3339, 3257, 4466, 4539, 4529, 4581, 6728, 6786, 6746, 6737, 6643, 2724, 2313, 2425, 2536, 2653, 6186, 5776, 5888, 5999, 6116]
    front_waist_10_vertex_idx = [411, 135, 179, 3647, 3693, 2877, 3015, 3078, 3076, 1893, 2891, 1393, 1666, 1736, 1591, 2230, 2106, 2150, 2176, 5334, 5312, 4867, 5087, 5205, 5060, 5691, 5567, 5611, 5637, 1449, 3503, 4921, 2911, 1784, 6370, 850, 1012, 1018, 1094, 3328, 3387, 3347, 3339, 3258, 4459, 4498, 4506, 4580, 6728, 6786, 6746, 6737, 6657, 2707, 2724, 2067, 2313, 2389, 2425, 2477, 2536, 2634, 2653, 6168, 6189, 5528, 5776, 5850, 5888, 5938, 5999, 6095, 6116]

    character_config = {
        'body': opts.config.strip('_' + opts.config.split('_')[-1]),
        'finger': opts.config.split('_')[-1]
    }

    if opts.config == 'production_5':
        marker_vertex_idx = production_5_vertex_idx
    elif opts.config == 'front_waist_10':
        marker_vertex_idx = front_waist_10_vertex_idx

    grab_file_info_path = os.path.join('data', 'GRAB_file_list.npz')
    cmu_file_info_path = os.path.join('data', 'CMU_file_list.npz')

    grab_file_info = np.load(grab_file_info_path)
    grab_file_folder = grab_file_info['file_folder']
    grab_file_name_list = grab_file_info['file_name_list']
    grab_files = [os.path.join(grab_file_folder[i], grab_file_name_list[i]) for i in range(len(grab_file_name_list))]

    cmu_file_info = np.load(cmu_file_info_path)
    cmu_file_folder = cmu_file_info['file_folder']
    cmu_file_name_list = cmu_file_info['file_name_list']
    cmu_files = [os.path.join(cmu_file_folder[i], cmu_file_name_list[i]) for i in range(len(cmu_file_name_list))]

    grab_motion_num = len(grab_files)
    cmu_motion_num = len(cmu_files)
    cmu_motion_set = np.random.permutation(cmu_motion_num)
    grab_motion_set = np.random.permutation(grab_motion_num)
    motion_set = []
    for i in range(cmu_motion_num):
        motion_set += [[cmu_motion_set[i], grab_motion_set[i % grab_motion_num]]]
    # motion_set = range(motion_num)

    grab_dataset_path = os.path.join(opts.dataset_dir, 'GRAB')
    cmu_dataset_path = os.path.join(opts.dataset_dir, 'CMU')

    # marker_idx = [i for i in range(len(marker_vertex_idx))]
    cuda = True
    random_betas = False
    batch_size = 100

    smplh_model = smplx.create(
        model_path=os.path.join('external', 'smplx', 'model'),
        model_type='smplh', gender='neutral',
        num_betas=16, ext='pkl',
        batch_size=batch_size,
        use_pca=False
    )
    print(smplh_model)

    for cmu_motion, grab_motion in tqdm(motion_set[:]):
        try:
            cmu_motion_data = np.load(os.path.join(cmu_dataset_path, cmu_files[cmu_motion]))
            cmu_poses = cmu_motion_data['poses']
            betas = cmu_motion_data['betas']
            trans = cmu_motion_data['trans']
            grab_motion_data = np.load(os.path.join(grab_dataset_path, grab_files[grab_motion]))
            grab_poses = grab_motion_data['poses']
        except:
            # print('Could not read %s ! skipping..' % cmu_files[cmu_motion] + grab_files[grab_motion])
            continue
        poses = merge_motion(cmu_poses, grab_poses)
        N = len(poses)
        total_frame += N
        betas = betas.astype(np.float32)
        poses = poses.astype(np.float32)
        trans = trans.astype(np.float32)
        betas = torch.tensor(betas)
        poses = torch.tensor(poses)
        trans = torch.tensor(trans)
        if random_betas:
            betas = torch.randn([1, 16], dtype=torch.float32)
        else:
            betas = betas.reshape(1, 16)

        root_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
        left_hand_pose = poses[:, 66:111]
        right_hand_pose = poses[:, 111:]

        M = np.zeros((N, len(marker_vertex_idx), 3), dtype=np.float32)
        M1 = np.zeros((N, len(marker_vertex_idx), 3), dtype=np.float32)
        J_R = np.zeros((N, 52, 3, 3), dtype=np.float32)
        J_t = np.zeros((N, 73, 3), dtype=np.float32)
        n_batch = int(np.ceil(N / batch_size))

        for i in range(n_batch):
            root_orient_batch = root_orient[batch_size * i: min(batch_size * (i + 1), N)]
            betas_batch = betas.repeat((root_orient_batch.shape[0], 1))
            body_pose_batch = body_pose[batch_size * i: min(batch_size * (i + 1), N)]
            left_hand_pose_batch = left_hand_pose[batch_size * i: min(batch_size * (i + 1), N)]
            right_hand_pose_batch = right_hand_pose[batch_size * i: min(batch_size * (i + 1), N)]
            trans_batch = trans[batch_size * i: min(batch_size * (i + 1), N)]
            if cuda:
                root_orient_batch = root_orient_batch.cuda()
                betas_batch = betas_batch.cuda()
                body_pose_batch = body_pose_batch.cuda()
                left_hand_pose_batch = left_hand_pose_batch.cuda()
                right_hand_pose_batch = right_hand_pose_batch.cuda()
                trans_batch = trans_batch.cuda()
                smplh_model.cuda()

            batch_output = smplh_model(
                betas=betas_batch,
                global_orient=root_orient_batch,
                body_pose=body_pose_batch,
                left_hand_pose=left_hand_pose_batch,
                right_hand_pose=right_hand_pose_batch,
                transl=trans_batch,
                return_verts=True,
                return_full_pose=True
            )
            marker_vertices = batch_output.vertices.cpu().detach().numpy()
            M[batch_size * i: min(batch_size * (i + 1), N), :, :] = marker_vertices[:, marker_vertex_idx,:]
            batch_full_pose = batch_rodrigues(batch_output.full_pose.view(-1, 3)) \
                .view([batch_output.full_pose.shape[0], -1, 3, 3])
            J_R[batch_size * i: min(batch_size * (i + 1), N), :, :, :] = \
                batch_full_pose.cpu().detach().numpy().squeeze()
            batch_joints = batch_output.joints.cpu().detach().numpy().squeeze()
            J_t[batch_size * i: min(batch_size * (i + 1), N), :, :] = batch_joints

        output_file_folder = '{}/{}_CMU+GRAB_{}'.format(opts.output_dir, opts.config, opts.save_name)
        if not os.path.exists(output_file_folder):
            os.makedirs(output_file_folder)
        output_file_folder = os.path.join(output_file_folder, cmu_file_folder[cmu_motion])
        if not os.path.exists(output_file_folder):
            os.mkdir(output_file_folder)
        output_file = os.path.join(output_file_folder, cmu_file_name_list[cmu_motion] +
                                   grab_file_folder[grab_motion] + '_' + grab_file_name_list[grab_motion])

        weights1 = smplh_model.lbs_weights[marker_vertex_idx, :].cpu().numpy()
        # marker_config = get_marker_config(t_pose_marker, Joint, weights1)
        M1, corrupted_marker_vis = corrupt_occ(character_config, M, opts.body_occ_rate, opts.finger_occ_rate)
        M1 = corrupt_shift(character_config, M1, opts.body_shift_prob, opts.finger_shift_prob,
                           opts.body_shift_scale, opts.finger_shift_scale)

        # get t-pose
        v_shaped = smplh_model.v_template + smplx.lbs.blend_shapes(betas.cuda(), smplh_model.shapedirs)
        Joint = smplx.lbs.vertices2joints(smplh_model.J_regressor, v_shaped).cpu().squeeze().numpy()
        t_pose_marker = np.array(v_shaped.cpu().detach().squeeze())[marker_vertex_idx, :]

        trans = trans.numpy() + Joint[0]
        t_pose_marker -= Joint[0]
        Joint = Joint - Joint[0]

        J_t = J_t * 100
        M = M * 100
        M1 = M1 * 100
        t_pose_marker = t_pose_marker * 100
        trans = trans * 100
        Joint = Joint * 100

        np.savez(output_file, marker=M, marker_corrupted=M1, corrupted_marker_vis=corrupted_marker_vis,
                 joint_pos=J_t, joint_rot=J_R, trans=trans, parents=smplh_model.parents.cpu().numpy(),
                 shape=betas.numpy(), t_pose_joint_pos=Joint, t_pose_marker_pos=t_pose_marker,
                 weights=weights1)
        data_list += [{
            'merged': os.path.join(cmu_file_folder[cmu_motion], cmu_file_name_list[cmu_motion] +
                                   grab_file_folder[grab_motion] + '_' + grab_file_name_list[grab_motion]),
            'cmu': cmu_files[cmu_motion],
            'grab': grab_files[grab_motion],
            'body_occ_rate': opts.body_occ_rate,
            'finger_occ_rate': opts.finger_occ_rate,
            'body_shift_prob': opts.body_shift_prob,
            'finger_shift_prob': opts.finger_shift_prob,
            'body_shift_scale': opts.body_shift_scale,
            'finger_shift_scale': opts.finger_shift_scale
        }]
    with open('{}/{}_CMU+GRAB_{}/data_list.json'.format(opts.output_dir, opts.config, opts.save_name), 'w') as f:
        json.dump(data_list, fp=f, indent='    ')


if __name__ == '__main__':
    set_seed(opts.seed)
    generate_data()

# Description
This repo is for generating synthetic dataset. We merge the body motion from CMU motion dataset and the hand motion from GRAB motion dataset, and introduce marker occlusion and shifting to generate an augmented synthetic dataset. Both the data of CMU and GRAB come from AMASS, and the motions are skinned to mesh with SMPL+H.


# Setup
## 1. Install SMPLX
```
cd external
git clone https://github.com/vchoutas/smplx
cd smplx
python setup.py install
```
## 2. Prepare SMPL+H model
We use SMPL+H model to generate mesh, which can be downloaded [here](https://mano.is.tue.mpg.de). Please download both the models from "Extended SMPL+H model (used in AMASS project)" and "Models & Code". Extract the downloaded files in `external` and you should get `external/smplh` and `external/mano_v1_2`. Then merge these models. Here is the command:
```
cd external/smplx
mkdir model
mkdir model/smplh

python tools/merge_smplh_mano.py \
--smplh-fn ../smplh/neutral/model.npz \
--mano-left-fn ../mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn ../mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder ./model/smplh

mv ./model/smplh/model.pkl ./model/smplh/SMPLH_NEUTRAL.pkl
```
Here we take neutral model as example. Please merge the models of other genders if you need them.

More details about SMPLX module and SMPL+H model, please see this [repository](https://github.com/vchoutas/smplx).

## 3. Prepare dataset
Download the CMU dataset and GRAB dataset from [AMASS](https://amass.is.tue.mpg.de and extract them in the `data/AMASS` ,and the folder structure of `data` is like follows:

```
.
├── AMASS
│   ├── CMU
│   │   ├── 01
│   │   ├── ...
│   └── GRAB
│       ├── LICENSE.txt
│       ├── s1
│       ├── ...
├── CMU_file_list.npz
├── gap_info
├── generated
└── GRAB_file_list.npz

```

The full file lists of these two datasets have been put in `./data` as `CMU_file_list.npz` and `GRAB_file_list.npz`. If you decide not to use the full dataset, you can generate your own file lists using `file_list_script.py`.

## 4. Generate synthetic dataset
Finally, you can generate augmented synthetic data using `generate_train_data_merge.py`:
```
python generate_train_data_merge.py \
--body_occ_rate 0.1 \
--finger_occ_rate 0.1 \
--body_shift_prob 0.001 \
--finger_shift_prob 0.001 \
--body_shift_scale 0.25 \
--finger_shift_scale 0.1 \
--save_name example \
--output_dir data/generated \
--dataset_dir data/AMASS \
--config=front_waist_10
```

Every generated file contain following data:
* `marker`: positions of all markers (clean)
* `marker_corrupted`: positions of all markers (clean)
* `corrupted_marker_vis`: visible matrix of corrupted markers
* `joint_pos`: positions of all joints
* `joint_rot`: rotation matrix of all joints
* `trans`: transition of the root
* `parents`: parent tree of the smplh skeleton
* `shape`: shape parameter(beta) of the character
* `t_pose_joint_pos`: positions of all joints in T-pose
* `t_pose_marker_pos`: positions of all markers in T-pose

Note that the unit of shift_scale is meter. The generation process will take for about 30-60 minutes.

After generation process is done, there will be a json file of the configurations of all generated files in the output_dir.
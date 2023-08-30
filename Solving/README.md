# Description

This folder contains the code and sample data for occlusion fixing and motion solving.

## Occlusion Sixing

Check out  `preprocess.py` to run occlusion fixing using EDM algorithm and neural network. You can change the input and output file by modifying lines 104-106

```python
    IN_FILE_PATH = "SampleData/32_01_poses.npzs10_spheresmall_pass_1_stageii.npz"
    OUT_FILE_PATH = "SampleData/32_01_poses.npzs10_spheresmall_pass_1_stageii_fixed.npz"
    main(args, IN_FILE_PATH, OUT_FILE_PATH)
```

## Motion Solving

Check out  `process.py` to solve motions using the heterogeneous graph neural network. You can change the input and output file by modifying lines 104-106

```python
    IN_FILE_PATH = "SampleData/32_01_poses.npzs10_spheresmall_pass_1_stageii_fixed.npz"
    OUT_FILE_PATH = "SampleData/network_32_01_poses.npzs10_spheresmall_pass_1_stageii.bvh"
    main(args, IN_FILE_PATH, OUT_FILE_PATH)
```

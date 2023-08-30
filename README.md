# Description
This repo is for generating synthetic dataset. We merge the body motion from CMU motion dataset and the hand motion from GRAB motion dataset, and introduce marker occlusion and shifting to generate an augmented synthetic dataset. Both the data of CMU and GRAB come from AMASS, and the motions are skinned to mesh with SMPL+H.

## Setup

````
conda env create --name LocalMoCap -f env.yaml
conda activate LocalMoCap
````



## Synthetic Data Generation

Please check the README file in the subfolder `DataSynthesis`.

## Motion Solving

Please check the README file in the subfolder `Solving`.

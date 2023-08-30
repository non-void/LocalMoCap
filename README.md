# **A Locality-Based Neural Solver for Optical Motion Capture**
![representative](imgs/representative.jpg)

## Abstract

We present a novel locality-based learning method for cleaning and solving optical motion capture data. Given noisy marker data, we propose a new heterogeneous graph neural network which treats markers and joints as different types of nodes, and uses graph convolution operations to extract the local features of markers and joints and transform them to clean motions. To deal with anomaly markers (e.g. occluded or with big tracking errors), the key insight is that a marker’s motion shows strong correlations with the motions of its immediate neighboring markers but less so with other markers, a.k.a. locality, which enables us to efficiently fill missing markers (e.g. due to occlusion). Additionally, we also identify marker outliers due to tracking errors by investigating their acceleration profiles. Finally, we propose a training regime based on representation learning and data augmentation, by training the model on data with masking. The masking schemes aim to mimic the occluded and noisy markers often observed in the real data. Finally, we show that our method achieves high accuracy on multiple metrics across various datasets. Extensive comparison shows our method outperforms state-of-the-art methods in terms of prediction accuracy of occluded marker position error by approximately 20%, which leads to a further error reduction on the reconstructed joint rotations and positions by 30%. The code and data for this paper are available at github.com/localmocap/LocalMoCap .

[[Project website](http://www.cad.zju.edu.cn/home/jin/SigA20231/NeuralSolver.htm)]


## Setup

````
conda env create --name LocalMoCap -f env.yaml
conda activate LocalMoCap
````

## Synthetic Data Generation

Please check the README file in the subfolder `DataSynthesis`.

## Motion Solving

Please check the README file in the subfolder `Solving`.

## Cite


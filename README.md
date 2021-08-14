# Learning a Disentangled Representation for Human Pose Forecasting

## _Absract_:

Human pose forecasting, \em{i.e.}, forecasting human body keypoints' locations given a sequence of observed ones, is a challenging task due to the uncertainty in human pose dynamics. 
Many approaches have been proposed to solve this problem, including Long Short-Term Memories (LSTMs) and Variational AutoEncoders (VAEs). Yet, they do not effectively predict human motions when both global trajectory and local pose movements exist.
We propose to learn a representation that disentangles the global and local pose forecasting tasks. We also show that it is better to stop the prediction when the uncertainty in human motion increases. 
Our forecasting model outperforms all existing methods on the pose forecasting benchmark to date by over $20\%$. The code will be made available online.

## Introduction:
This is the official code for the paper ["Learning a Disentangled Representation for Human Pose Forecasting"](link), accepted and published in [ICCV 2021](http://iccv2021.thecvf.com/home)

## Contents
------------
  * [Repository Structure](#repository-structure)
  * [Proposed Method](#proposed-method)
  * [Results](#results)
  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Training/Testing](#training-testing)
  * [Tested Environments](#tested-environments)
  
## Repository structure:
------------
    ├── pose-prediction                 : Project repository
          ├── 3dpw 
            ├── train.py                : Script for training.  
            ├── test.py                 : Script for testing.  
            ├── DataLoader.py           : Script for data loader. 
            ├── model.py                : Script containing the implementation of the network.
            ├── utils.py                : Script containing necessary functions.
          ├── posetrack
            ├── train.py                : Script for training.  
            ├── test.py                 : Script for testing.  
            ├── DataLoader.py           : Script for data loader. 
            ├── model.py                : Script containing the implementation of the network.
            ├── utils.py                : Script containing necessary functions.
            
## Proposed method
-------------
![Our proposed method](images/network.pdf)

![Our proposed network architecture for trajectory prediction](Images/fig2.png)

![Our proposed network architecture for pose prediction](Images/fig3.png)




## Results
--------------
![Comparision of our model with other methods](images/tab.png)
![Example of outputs](Images/fig4-a.png.png)
![Example of outputs](Images/fig4-b.png.png)
  
## Installation:
------------
Start by cloning this repositiory:
```
git clone https://github.com/vita-epfl/bounding-box-prediction.git
cd bounding-box-prediction
```
Create a new conda environment (Python 3.7):
```
conda create -n pv-lstm python=3.7
conda activate pv-lstm
```
And install the dependencies:
```
pip install -r requirements.txt
```

## Dataset:
  
  * Clone the dataset's [repository](https://github.com/ykotseruba/JAAD).
  ```
  git clone https://github.com/ykotseruba/JAAD
  ```
  * Run the `prepare_data.py` script, make sure you provide the path to the JAAD repository and the train/val/test ratios (ratios must be in [0,1] and their sum should equal 1.
  ```
  python3 prepare_data.py |path/to/JAAD/repo| |train_ratio| |val_ratio| |test_ratio|
  ```
  * Download the [JAAD clips](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) (UNRESIZED) and unzip them in the `videos` folder.
  * Run the script `split_clips_to_frames.sh` to convert the JAAD videos into frames. Each frame will be placed in a folder under the `scene` folder. Note that this takes 169G of space.
  
  
## Training/Testing:
Open `train.py` and `test.py` and change the parameters in the args class depending on the paths of your files.
Start training the network by running the command:
```
python3 train.py
```
Test the trained network by running the command:
```
python3 test.py
```

## Tested Environments:
------------
  * Ubuntu 18.04, CUDA 10.1
  * Windows 10, CUDA 10.1

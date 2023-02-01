# AENC: Adaptive Neural Ensemble Controller for Mitigating Latency Problems in Vision-based Autonomous Driving

## Introduction
This project studies the latency problem on vision-based neural network autonomous driving in the lane-keeping task and proposes a vision-based novel neural network controller, the Adaptive Neural Ensemble Controller (ANEC), that is inspired by the near/far gaze distribution of human drivers during lane-keeping. 
We conducted the experiments on OSCAR simulator that is user-friendly and customizable. We updated the OSCAR platform to enable the
simultaneous operation of several neural network controllers because it was originally designed to support just one neural network controller.

## How does it work?
We propose the Adaptive Neural Ensemble Controller (ANEC), by which we show algorithmic perception latency problems can be mitigated by adaptively infusing the prediction action output into the baseline output. A high-level overview of the proposed system is shown in the Figure below. ANEC depends on combining the output of two driving models, the Base Model (BM) and the Predictive Model (PM). BM is expected to extract features to infer control output at the current input without consideration of latency. PM, on the other hand, is expected to extract latent variables for future actions. A dy- namic and adaptive weight, dependent on the vehicle speed, is assigned to each model to establish ANEC final output. The higher the speed, the greater the significance of future states, and hence the greater the weight assigned to PM.
![My Image](../images/my-image.jpg)


## General steps to use ANEC:
1. Download ANEC source code.
     * Understant the directory structure.
2. Create the conda environment.
     * Activate the environment
3. Start OSCAR with the fusion vehicle.
4. Collect data, understand its format, and clean it if needed.
    * This will create a CSV file which will be used for the base model (BM) training.
5. Reassociate data
    * This will create a 2nd CSV file which will be used for the predictive model (PM) training.
6. Train the BM
7. Train the PM
8. 


## Download the OSCAR_ANEC Source Code

```
$ git clone --branch devel_anec https://github.com/jrkwon/oscar.git --recursive 
```

## Directory Structure
- `catkin_ws`: ros workspace
  - `src`
    - `data_collection`: data from front camera and steering/throttle
    - `fusion`: Ford Fusion Energia model
    - `rover`: 
- `config`: configurations
  - `conda`: conda environment files
  - `config.yaml`: config file names for neural_net, data_collection, and run_neural
  - `neural_net`: system settings for neural_net
  - `data_collection`: system settings for data_collection
  - `run_neural`: system settings for run_neural
- `neural_net`: neural network package for end to end learning
- `PX4-Autopilot`: The folder for the PX4 Autopilot.

## Prior to Use

### Versions 

The OSCAR has been tested with ROS Melodic on Ubuntu 18.04.

### Install ROS packages
Install two more packages for this project unless you already have them in your system.
```
$ sudo apt install ros-$ROS_DISTRO-fake-localization
$ sudo apt install ros-$ROS_DISTRO-joy

```

### Create Conda Environment 

Create a conda environment using an environment file that is prepared at `config/conda`.
```
$ conda env create --file config/conda/ANEC_env.yaml
```

## How to Use

### Activate Conda Environment

Activate the `anec` environment. 
```
$ conda activate anec
```

ANEC was tested using 'fusion' vehicle only, 'rover' was not tested.
This section explains how to use `fusion`.

### fusion

`fusion` is heavily relied on OSRF's `car_demo` project. Simply use the following script.

```
(anec) $ ./start_fusion.sh track_aws_ANEC
```

## How to Collect Data

Run the script with a data ID as an argument.
```
(anec) $ ./collect_data_fusion jaerock
```

The default data folder location is `$(pwd)e2e_{fusion/rover}_data`.

### Data Format
ANEC was developed using a version of OSCAR after 0.92. previous versions did not have a column for 'brake'.Use `convert_csv.py` to convert a data CSV file collected before 0.92 to a new CSV file.

#### From Version 0.92

Data Collection will save a csv file with images. The CSV file has following columns

```
image_file_name / steering_angle / throttle / brake / linux_time / velocity / velocity_x / velocity_y / velocity_z / position_x / position_y / position_z

```

```
2020-12-08-23-55-31-150079.jpg	-0.0149267930537	0.15	0.7 1607489731.15	0.846993743317	0.846750728334	-0.00903874268025	-0.0181633261171	8.25840907119	-102.836707258	0.0248406100056

```

## Data Cleaning

When some of test images must be deleted, just delete them and rebuild the csv using `rebuild_csv.py`.

```
(anec) $ python rebuild_csv.py path/to/data/folder
```

## How to Train Neural Network

[ANEC-Training.pdf](https://github.com/jrkwon/oscar/files/10561225/ANEC-Training.pdf)

After collecting data, we will train each model separately.
* BM will be trained using the generated csv file after collecting data.
* We create a second csv file to train PM, where we modify the original csv file by associating the images with future actions


### steering_angle_scale

`steering_angle_scale` in `neural_net` config is for making the neural network have higher precision in prediction. The range of steering angle is -1 to 1. But in most cases, there will not be values between -1 and -0.5 as well as between 0.5 to 1 which means very sharp steering angles. These sharp steering angles will not be collected from driving a track in practice.

To find a proper scale value, you may use `test_data.py` by which you can see data statistics. The following is an example.

The choice of `steering_angle_scale` is especially important when activation functions are `sigmoid` or `tanh` in which you may lose data samples of sharp turns.

```
####### data statistics #########
Steering Command Statistics:
count    6261.000000
mean        0.002407
std         0.134601
min        -0.421035
25%        -0.016988
50%         0.009774
75%         0.085238
max         0.310105
Name: steering_angle, dtype: float64
```

### build a second csv file to create (image, future actions) data pairs.
After collecting data and cleaning it, you will have a generated csv file that can be used to train the base model (BM).
Now, we need a 2nd csv file to train the predictive model (PM), weher we need to create (image, future actions) data pairs.

This is done by running the following script with 2 arguments: 
  1. path/to/data/folder --> collected data path
  2. delta_value --> expected delay value (timesteps)
    * if delta is 5 then the image at time t will be paired with the action taken at time t+5
```
(anec) $ python neural_net/reassociate_data.py path/to/data/folder delta_value
```


### Training

1. Train the base model (BM) and save the trained model.
```
(anec) $ . setup.bash
(anec) $ python neural_net/train_anec.py BM path/to/data/folder
```

2. Train the predictive model (PM) and save the trained model.
```
(anec) $ . setup.bash
(anec) $ python neural_net/train_anec.py PM path/to/data/folder
```


## How to Drive using ANEC system 

```
(oscar) $ . setup.bash
(oscar) $ rosrun run_neural run_nerual_fusion.py -bm path/to/saved/base/model -pm path/to/saved/predictive/model
```


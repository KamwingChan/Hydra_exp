# Hydra_exp
## Introduction
An experiment of deploying Hydra on realsense D455, including semantic_inference and kimera_vio modules.

## Follow [Hydra](https://github.com/MIT-SPARK/Hydra) to finish quick start
```bash
cd ~/catkin_ws
catkin build kimera_vio_ros -j16
```
## Launch Method (In seperate terminal)
Using semantic_inference as semantic input (Mask2former segmenter is under develop)
```bash
roslaunch hydra_ros realsense.launch model_name:=ade20k-hrnetv2-c1
roslaunch hydra_ros realsense.launch model_name:=ade20k-mask2former-r50
roslaunch realsense2_camera hydra_realsense.launch
roslaunch kimera_vio_ros kimera_vio_d455.launch 
```
## Future Work
LLM, More physical information...

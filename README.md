# Hydra_exp
## Introduction
An experiment of deploying Hydra on realsense D455, including semantic_inference and kimera_vio modules.

## Follow [Hydra](https://github.com/MIT-SPARK/Hydra) to finish quick start
```bash
cd ~/catkin_ws
git clone https://github.com/KamwingChan/Hydra_exp.git
catkin build kimera_vio_ros -j16
catkin build semantic_inference_ros
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
- [ ] cuvslam
- [ ] SpatialLM
- [ ] Material Perception

## Developer tools
### Using [mmlab](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md#supported-models)
Learning [README.md编写方法](https://blog.csdn.net/weixin_49941024/article/details/147166930?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0-147166930-blog-129700568.235^v43^pc_blog_bottom_relevance_base6&spm=1001.2101.3001.4242.1&utm_relevant_index=2)
```bash
conda activate openmmlab
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

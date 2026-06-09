# Hydra_exp（PhyGraph v1）
## Introduction
An experiment of deploying Hydra on __Realsense D455__, including semantic_inference and kimera_vio modules.

__New__ (25.09)The "Continue Mapping" feature is currently in early development. Use with caution.

__New__ (25.11)A big update on Phy_graph, and Phy_plan is coming soon

__New__ (25.12)Available on [BEHAVIOR-1k](https://behavior.stanford.edu/index.html), check the [script](/phy_plan/env) in /phy_plan/env

__New__ (26.01)Phy_plan(alpha) is available now. But big new feature is comming soon.
## Follow [Hydra](https://github.com/MIT-SPARK/Hydra/tree/archive/ros_noetic) to finish quick start !!! Choose the ros1 branch!!!
```bash
cd catkin_ws/src
git clone -b archive/ros_noetic https://github.com/MIT-SPARK/Hydra.git
vcs import . < hydra/install/hydra.rosinstall
rosdep install --from-paths . --ignore-src -r -y
rm -rf hydra hydra_ros semantic_inference kimera_pgmo 
git clone https://github.com/KamwingChan/Hydra_exp.git
git clone -b ros1-legacy https://github.com/IntelRealSense/realsense-ros.git
catkin build
mv hydra_realsense.launch realsense-ros/realsense2_camera/launch
```
## Launch Method (In seperate terminal)
Using semantic_inference as semantic input (Mask2former segmenter is under develop)

Realsense example
```bash
roslaunch hydra_ros realsense.launch model_name:=ade20k-segformer-b5
roslaunch realsense2_camera hydra_realsense.launch
roslaunch kimera_vio_ros kimera_vio_d455.launch 
```

Isaaclab example (Forest Navigation)
```bash
roslaunch hydra_ros isaacsim.launch model_name:=ade20k-segformer-b5 sim_time_required:=true

//In seperate bash
conda activate <your isaaclab env>
python ./scripts/rsl_rl/keyboard.py \
    --task=Forestnavigation-Keyboard-Play-v0 \
    --enable_cameras \
    --enable-ros
```

Behavior example
```bash
roslaunch hydra_ros behavior.launch sim_time_required:=true

//In seperate bash
conda activate <your isaaclab env>
cd catkin_ws/src/phy_plan
python env/behavior_ros.py //check the argument in the file and use WASD to move
```

## Future Work
- [x] Room Classification(Needed Advanced)
- [x] Continue Mapping Mode
- [x] Physical information
- [ ] Phy_plan (on-going)
- [ ] Paper

## Physical imformation
### Our Method

- [ ] Scene Graph Based Planning
- [x] Enhanced Scene Graph via phy_graph

## Phy_plan: High-level task planning using phy_graph


## Developer tools
### Using [mmlab](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md#supported-models)
```bash
cd ${YOUR_WS}/mmdeploy
conda activate openmmlab
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
# Building ops for tensort (needed by semantic_inference)
mkdir build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=trt ..
make -j$(nproc)
```
Then you will see _libmmdeploy_tensorrt_ops.so_ in _lib/_.
```bash
sudo cp lib/libmmdeploy_tensorrt_ops.so /usr/lib/libmmdeploy_tensorrt_ops.so
```

### Using segformer provided by [MMsegment](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/mask2former)
```bash
cd ${YOUR_WS}/mmdeploy
wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth
python ./tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_static-480x640.py \
    ~/workspace/mmsegmentation/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py \
    segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth \
    ~/Mask2Former/1_Color.png \
    --work-dir mmdeploy_model/segformer \
    --device cpu \
    --show \
    --dump-info
```
Then move output model to semantic inference package. [Semantic inference](https://github.com/MIT-SPARK/semantic_inference) will automatically build engine for TensorRT.
```bash
mv $YOUR_WS/mmdeploy/mmdeploy_model/mask2former-ade20k-r50/end2end.onnx ~/catkin_ws/src/semantic_inference/semantic_inference/models/ade20k-mask2former-r50.onnx
roslaunch semantic_inference_ros semantic_inference.launch model_name:=ade20k-mask2former-r50
```

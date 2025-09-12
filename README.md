# Hydra_exp
## Introduction
Learning [README.md](https://blog.csdn.net/weixin_49941024/article/details/147166930?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0-147166930-blog-129700568.235^v43^pc_blog_bottom_relevance_base6&spm=1001.2101.3001.4242.1&utm_relevant_index=2).
An experiment of deploying Hydra on __Realsense D455__, including semantic_inference and kimera_vio modules.

## Follow [Hydra](/hydra/README.md) to finish quick start
```bash
cd catkin_ws/src
git clone https://github.com/KamwingChan/Hydra_exp.git
catkin build kimera_vio_ros -j16
catkin build semantic_inference_ros
mv hydra_realsense.launch realsense-ros/realsense2_camera/launch
```
## Launch Method (In seperate terminal)
Using semantic_inference as semantic input (Mask2former segmenter is under develop)
```bash
roslaunch hydra_ros realsense.launch model_name:=ade20k-segformer-b5
roslaunch realsense2_camera hydra_realsense.launch
roslaunch kimera_vio_ros kimera_vio_d455.launch 

```
## Future Work
- [ ] Room Classification
- [ ] SpatialLM
- [x] [Physical information](/physical_inference)
- [ ] Paper

## Physical imformation
### Our Method

- [ ] VLN
- [ ] SpatialLM
- [x] Material Perception

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

# Hydra_exp
## Introduction
Learning [README.md](https://blog.csdn.net/weixin_49941024/article/details/147166930?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0-147166930-blog-129700568.235^v43^pc_blog_bottom_relevance_base6&spm=1001.2101.3001.4242.1&utm_relevant_index=2)
An experiment of deploying Hydra on ==realsense D455== , including semantic_inference and kimera_vio modules.

## Follow [Hydra](https://github.com/MIT-SPARK/Hydra) to finish quick start
```bash
cd C
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
```bash
conda activate openmmlab
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```
### Using mask2former provided by [MMsegment](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/mask2former)
```bash
cd $YOUR_WS/mmdeploy
wget https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024/mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth
python ./tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_dynamic.py \
    ~/workspace/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py \
    mask2former_r50_8xb2-160k_ade20k-512x512_20221204_000055-2d1f55f1.pth \
    ~/Mask2Former/1_Color.png \
    --work-dir mmdeploy_model/mask2former-ade20k-r50 \
    --device cpu \
    --show \
    --dump-info
```
Then move output model to semantic inference package. [Semantic inference](https://github.com/MIT-SPARK/semantic_inference) will automatically build engine for TensorRT.
```bash
mv $YOUR_WS/mmdeploy/mmdeploy_model/mask2former-ade20k-r50/end2end.onnx ~/catkin_ws/src/semantic_inference/semantic_inference/models/ade20k-mask2former-r50.onnx
roslaunch semantic_inference_ros semantic_inference.launch model_name:=ade20k-mask2former-r50
```

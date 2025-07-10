# test_mask2former_onnx.py

import click
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch
import torch.nn.functional as F

# This is needed to load the config for preprocessing
import sys
sys.path.insert(0, str(Path.home() / "Mask2Former"))

from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.catalog import MetadataCatalog

@click.command()
@click.option(
    "--onnx-model",
    default="semantic_inference/models/mask2former_r50_ade20k_dynamic.onnx",
    help="Path to the ONNX model.",
)
@click.option(
    "--image",
    default="1_Color.png",
    help="Path to the input image for testing.",
)
@click.option(
    "--config-file",
    default="runs/segment/train_mask2former_detectron2/config.yaml",
    help="Path to the original model's config file for preprocessing info.",
)
@click.option(
    "--output-image",
    default="segmentation_test_result.png",
    help="Path to save the visualized segmentation result.",
)
def main(onnx_model, image, config_file, output_image):
    """
    Tests the exported Mask2Former ONNX model with dynamic input size.
    """
    # --- 1. Load ONNX model and create session ---
    print(f"Loading ONNX model from {onnx_model}...")
    try:
        ort_session = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print("Please ensure onnxruntime-gpu is installed and a compatible CUDA version is available.")
        return
        
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    print(f"Input name: {input_name}, Output names: {output_names}")

    # --- 2. Load and Preprocess Image ---
    print(f"Loading and preprocessing image: {image}")
    
    cfg = get_cfg()

    # Add Mask2Former specific configs
    from mask2former.config import add_maskformer2_config
    add_maskformer2_config(cfg)

    # Load the base config file, which defines the MODEL.RESNETS node
    base_config_path = str(Path("~/Mask2Former/configs/ade20k/semantic-segmentation/Base-ADE20K-SemanticSegmentation.yaml").expanduser())
    cfg.merge_from_file(base_config_path)

    # Manually set default values for keys that exist in the training config but not in the base config.
    # This prevents the KeyError.
    # --- RESNETS ---
    cfg.MODEL.RESNETS.DEFORM_MODULATED = False
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]
    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    # NORM is in the base config, so we don't need to set it.

    # --- SEM_SEG_HEAD ---
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    # Other SEM_SEG_HEAD keys are already defined in add_maskformer2_config

    # --- SOLVER ---
    # Manually define missing solver keys
    if "POLY_LR_CONSTANT_ENDING" not in cfg.SOLVER:
        cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    if "POLY_LR_POWER" not in cfg.SOLVER:
        cfg.SOLVER.POLY_LR_POWER = 0.9
    if "BASE_LR_END" not in cfg.SOLVER:
        cfg.SOLVER.BASE_LR_END = 0.0
    if "NUM_DECAYS" not in cfg.SOLVER:
        cfg.SOLVER.NUM_DECAYS = 3
    if "REFERENCE_WORLD_SIZE" not in cfg.SOLVER:
        cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    if "RESCALE_INTERVAL" not in cfg.SOLVER:
        cfg.SOLVER.RESCALE_INTERVAL = False
    if "WEIGHT_DECAY_BIAS" not in cfg.SOLVER:
        cfg.SOLVER.WEIGHT_DECAY_BIAS = None # Use None for null in yaml
    if "WEIGHT_DECAY_EMBED" not in cfg.SOLVER:
        cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    
    # Now, merge the specific training config
    print(f"Loading specific training configuration from {config_file}...")
    cfg.merge_from_file(config_file)
    
    original_image = read_image(image, format="RGB")
    original_height, original_width = original_image.shape[:2]
    
    # The ONNX model was exported with a fixed size, so we resize the input image to match.
    # The export script uses (height, width) = (480, 640).
    aug = T.Resize((480, 640))
    transformed_image = aug.get_transform(original_image).apply_image(original_image)
    # Convert to tensor
    preprocessed_image_tensor = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
    
    # Normalize the image tensor for ONNX runtime
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
    normalized_image = (preprocessed_image_tensor - pixel_mean) / pixel_std
    # The ONNX model expects a (N, C, H, W) input.
    onnx_input = normalized_image.unsqueeze(0).numpy()

    # --- 3. Run Inference ---
    print(f"Running inference on image of size {onnx_input.shape[2:]}...")
    try:
        outputs = ort_session.run(output_names, {input_name: onnx_input})
        sem_seg_output = outputs[0]
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        return
    
    # --- 4. Post-process and Visualize ---
    print("Post-processing and visualizing the result...")
    
    # The output from the ONNX model is (N, num_classes, H, W).
    sem_seg_tensor = torch.from_numpy(sem_seg_output)
    sem_seg_resized = F.interpolate(
        sem_seg_tensor,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    # After interpolate, shape is (1, num_classes, H, W), so we can find the argmax.
    pred_classes = torch.argmax(sem_seg_resized, dim=1).squeeze(0).cpu()
    
    # Use Detectron2's Visualizer for better visualization
    # Need to register the dataset to get metadata
    from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
    MetadataCatalog.get("ade20k_sem_seg_val").set(
        stuff_classes=[c["name"] for c in ADE20K_150_CATEGORIES],
        stuff_colors=[c["color"] for c in ADE20K_150_CATEGORIES],
    )
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    visualizer = Visualizer(original_image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(pred_classes, alpha=0.5)
    
    # Save the result
    vis_output.save(output_image)
    print(f"Segmentation result saved to {output_image}")

if __name__ == "__main__":
    main()

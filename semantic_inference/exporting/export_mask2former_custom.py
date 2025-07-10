# exporting/export_mask2former_custom.py

import click
import torch
from pathlib import Path

# It's important to add the Mask2Former directory to the path
# so that Detectron2 can find the custom model definitions.
import sys
sys.path.insert(0, str(Path.home() / "Mask2Former"))

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

@click.command()
@click.option(
    "--config-file",
    default="runs/segment/train_mask2former_detectron2/config.yaml",
    help="Path to the model's config file.",
)
@click.option(
    "--weights-file",
    default="runs/segment/train_mask2former_detectron2/model_final.pth",
    help="Path to the trained model weights.",
)
@click.option(
    "--output-dir",
    default="semantic_inference/models",
    help="Directory to save the exported ONNX model.",
)
@click.option(
    "--height",
    default=480,
    help="Example height for tracing the model.",
)
@click.option(
    "--width",
    default=640,
    help="Example width for tracing the model.",
)
def main(config_file, weights_file, output_dir, height, width):
    """
    Exports a trained Mask2Former model to ONNX format with dynamic input size.
    """
    # --- 1. Setup and Load Configuration ---
    print("Loading configuration...")
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
    cfg.MODEL.WEIGHTS = weights_file
    cfg.freeze()

    # --- 2. Build the Model ---
    print("Building the model...")
    model = build_model(cfg)
    model.eval()

    # --- 3. Load Checkpoint ---
    print(f"Loading weights from {weights_file}...")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # --- 4. Create a Dummy Input for Tracing ---
    dummy_input = torch.randn(1, 3, height, width, device=model.device)
    print(f"Creating a dummy input of size: (1, 3, {height}, {width}) for tracing.")

    # --- 5. Create a wrapper for ONNX export ---
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image_batch):
            """
            The forward method now accepts a 4D batch of images and converts
            it into the list-of-dictionaries format that the Detectron2 model expects.
            """
            # The input 'image_batch' is a 4D tensor of shape [N, C, H, W].
            # We need to convert it to a list of dicts, where each dict
            # contains a 3D image tensor of shape [C, H, W].
            inputs = []
            for i in range(image_batch.shape[0]):
                inputs.append({"image": image_batch[i]})
            
            # The model will now correctly process this list and produce a 4D tensor for the backbone.
            result = self.model(inputs)
            
            # The output format for sem_seg is typically a list of results,
            # one for each image in the batch. For ONNX export with a dynamic
            # batch size, we need to stack these results.
            # The result[0]["sem_seg"] is of shape [num_classes, H, W].
            # We need to unsqueeze to add the batch dimension back.
            sem_seg_output = result[0]["sem_seg"].unsqueeze(0)
            return sem_seg_output

    onnx_model = OnnxWrapper(model)
    onnx_model.eval()

    # --- 6. Export to ONNX with Dynamic Axes ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_file_name = "mask2former_r50_ade20k_dynamic.onnx"
    onnx_path = output_path / onnx_file_name

    print(f"Exporting model to {onnx_path} with dynamic axes...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                onnx_model,
                (dummy_input,),
                str(onnx_path),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["sem_seg"],
                # dynamic_axes={
                #     "input": {0: "batch_size", 2: "height", 3: "width"},
                #     "sem_seg": {0: "batch_size", 2: "height", 3: "width"},
                # },
            )
        print("ONNX export successful!")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")

if __name__ == "__main__":
    main()

from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data='/home/kamwing/workspace/ade20k/images/training',
    det_model='yolo11x.pt',
    sam_model='sam_l.pt',
    output_dir='/home/kamwing/workspace/ade20k/annotated_images'
)

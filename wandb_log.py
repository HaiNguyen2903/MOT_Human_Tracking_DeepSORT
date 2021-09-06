import wandb

run = wandb.init(project = 'YOLOv5', tags = ['log inference videos'])

run.name = 'log inference videos'

infer_old = wandb.Artifact('detect_v2_reid_v2', type='Inference_videos')

infer_old.add_dir('/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v2_reid_v2')

run.log_artifact(infer_old)


infer_new = wandb.Artifact('detect_v3_reid_v5', type='Inference_videos')

infer_new.add_dir('/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v5')

run.log_artifact(infer_new)
import wandb

run = wandb.init(project = 'YOLOv5', tags = ['log ReID checkpoints'])

run.name = 'log ReID checkpoint'

# infer_old = wandb.Artifact('detect_v2_reid_v2', type='Inference_videos')

# infer_old.add_dir('/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v2_reid_v2')

# run.log_artifact(infer_old)


infer_new = wandb.Artifact('30ep_cbdatav5_lr_3e-4_decay_half_5ep_from_reid_v1_best_acc_eval_trajectory', type='ReID checkpoints')

infer_new.add_file('/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/deep_sort_pytorch/deep_sort/deep/checkpoint/cbdatav5/30ep_cbdatav5_lr_3e-4_decay_half_5ep_from_reid_v1_best_acc_eval_trajectory.t7')

run.log_artifact(infer_new)
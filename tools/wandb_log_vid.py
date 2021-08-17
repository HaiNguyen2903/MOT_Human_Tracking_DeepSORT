import wandb
from wandb import wandb_agent

run = wandb.init(project='demo_inference')

artifact = wandb.Artifact('new_demo_vid_2', type='demo_vids')

artifact.add_dir('/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/')

run.log_artifact(artifact)

# import wandb
# run = wandb.init()
# artifact = run.use_artifact('hainguyen/YOLOv5/run_3gqwg2vr_model:v0', type='model')
# artifact_dir = artifact.download()
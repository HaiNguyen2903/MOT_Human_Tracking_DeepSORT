search_dir='/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/test'

# infer_root_1='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v2_reid_v2'
# infer_root_2='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v5'
infer_root='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v8/mot_preds'

detect_v2='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/yolov5/runs/train/exp/weights/best.pt'
detect_v3='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/yolov5/runs/train/15epochs_640x640_b16/weights/best.pt'

reid_v2='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
reid_v5='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_datav1_lr_3e-4_decay_half_5ep_eval_trajectory.t7'

reid_v6='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_default.t7'
reid_v7='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_frame.t7'
reid_v8='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_trajectory.t7'


for vid_path in "$search_dir"/*.mp4
    do
        # basename "$vid_path"
        name="$(basename -- $vid_path)"
        # python track_video.py --source $vid_path --output $infer_root_1 --yolo_weights $detect_v2 --deep_sort_weights $reid_v2 --save-txt
        # python track_video.py --source $vid_path --output $infer_root_2 --yolo_weights $detect_v3 --deep_sort_weights $reid_v5 --save-txt
        python track_video.py --source $vid_path --output $infer_root --yolo_weights $detect_v3 --deep_sort_weights $reid_v8 --save-txt
    done
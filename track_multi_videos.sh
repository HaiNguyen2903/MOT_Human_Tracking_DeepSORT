
# search_dir='/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/test'

# search_dir on 1080Ti
search_dir='/data/DATA_ROOT/vtx_test_vids_16_10_21/vtx_video'

# infer_root_1='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v2_reid_v2'
# infer_root_2='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v5'

infer_root_default='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v8/mot_preds_default'
infer_root_frame='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v8/mot_preds_frame'
infer_root_trajectory='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference/detect_v3_reid_v8/mot_preds_trajectory'

infer_root='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference_tracking'

detect_v2='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/yolov5/runs/train/exp/weights/best.pt'
detect_v3='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/yolov5/runs/train/15epochs_640x640_b16/weights/best.pt'
detect_v4='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/yolov5/runs/train/15ep_cbdatav3_from_crowd_ckpt2/weights/best.pt'

reid_v2='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'

reid_v3='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_datav1_lr_3e-4_decay_half_5ep_eval_default.t7'
reid_v4='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_datav1_lr_3e-4_decay_half_5ep_eval_frame.t7'
reid_v5='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_datav1_lr_3e-4_decay_half_5ep_eval_trajectory.t7'

reid_v6='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_default.t7'
reid_v7='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_frame.t7'
reid_v8='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_trajectory.t7'



# detect v4 and reid v8 on 1080Ti

# detect_new='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/yolov5/runs/train/15ep_cbdatav3_from_crowd_ckpt2/weights/best.pt'
# reid_new='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/deep_sort_pytorch/deep_sort/deep/checkpoint/30ep_cbdatav2_lr_3e-4_decay_half_5ep_eval_trajectory.t7'

detect_crowd="/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/yolov5/checkpoints/crowdhuman_yolov5m_without_optimizer.pt"
reid_v2="/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"

detect_v5="/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/yolov5/runs/train/15epochs_640_640_cbdatav54/weights/best.pt"
reid_v10="/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/deep_sort_pytorch/deep_sort/deep/checkpoint/cbdatav5/30ep_cbdatav5_lr_3e-4_decay_half_5ep_from_reid_v1_best_acc_eval_trajectory.t7"

vid_output_old='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/inference_tracking/detect_crowd_reid_v2_on_cbdatav5/videos'
pred_output_old='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/inference_tracking/detect_crowd_reid_v2_on_cbdatav5/mot_preds'


vid_output_new='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/inference_tracking/infer_for_vtx_old_model/videos'
pred_output_new='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/inference_tracking/infer_for_vtx_old_model/mot_preds'

for vid_path in "$search_dir"/*.avi
    do
        # basename "$vid_path"
        name="$(basename -- $vid_path)"
        python track_video.py --source $vid_path --output $vid_output_new --yolo_weights $detect_crowd --deep_sort_weights $reid_v2 --save-vid
        python track_video.py --source $vid_path --output $pred_output_new --yolo_weights $detect_crowd --deep_sort_weights $reid_v2 --save-txt
        
        
        # python track_video.py --source $vid_path --output $infer_root_1 --yolo_weights $detect_v2 --deep_sort_weights $reid_v2 --save-txt
        # python track_video.py --source $vid_path --output $infer_root_2 --yolo_weights $detect_v3 --deep_sort_weights $reid_v5 --save-txt

        # python track_video.py --source $vid_path --output $infer_root/detect_v3_reid_v3/mot_preds --yolo_weights $detect_v3 --deep_sort_weights $reid_v3 --save-txt
        # python track_video.py --source $vid_path --output $infer_root/detect_v3_reid_v4/mot_preds --yolo_weights $detect_v3 --deep_sort_weights $reid_v4 --save-txt
        # python track_video.py --source $vid_path --output $infer_root/detect_v3_reid_v5/mot_preds --yolo_weights $detect_v3 --deep_sort_weights $reid_v5 --save-txt
        
        # python track_video.py --source $vid_path --output $infer_root_default --yolo_weights $detect_v3 --deep_sort_weights $reid_v6 --save-txt
        # python track_video.py --source $vid_path --output $infer_root_frame --yolo_weights $detect_v3 --deep_sort_weights $reid_v7 --save-txt
        # python track_video.py --source $vid_path --output $infer_root_trajectory --yolo_weights $detect_v3 --deep_sort_weights $reid_v8 --save-txt
    done
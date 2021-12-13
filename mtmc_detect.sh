search_dir='/data.local/all/hainp/MTMC_DATA/mp4_data'
weight='/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/yolov5/runs/train/15epochs_640_640_cbdatav54/weights/best.pt'

for vid_path in "$search_dir"/*.mp4
    do
        # basename "$vid_path"
        name="$(basename -- $vid_path)"
        python yolov5/detect.py --source $vid_path --weights $weight --save-txt --save-conf --save-crop --name "detect_for_mtmc"
    done
model_dir='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/deep_sort_pytorch/deep_sort/deep/checkpoint/cbdatav3'

features_dir='/data/hain/code/yolov5_deepsort/YoloV5_Deepsort_Clone/deep_sort_pytorch/deep_sort/deep/predicts'

data_version="$(basename -- $model_dir)"

save_dir=$features_dir/$data_version

mkdir -p $save_dir

for model in $model_dir/*;
    do 
        model_version="$(basename -- $model)"
        python test.py --ckpt $model --save-dir $save_dir --save-name ${model_version:0:-3}_from_reid_v2 --data-dir /data/DATA_ROOT/uet_reid
    done

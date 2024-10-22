# Additional running tutorial

## Data Generating
For faster and easier generating process, we first design a `MAIN_DATA_TREE` like below:

```bash
ROOT
|
|___data_version1
|	|
|	|__TRAIN_DATASET
|       |       |
|       |       |__video_name_1.mp4
|       |       |__video_name_1
|       |       |       |
|       |       |       |__gt
|       |       |           |__gt.txt
|       |       |           |__labels.txt
|       |       |
|       |       |
|       |       |__video_name_2.mp4
|       |       |__video_name_2
|       |               |
|       |               |__gt
|       |                   |__gt.txt
|       |                   |__labels.txt
|       |
|	|
|	|__detection_dataset
|	|		|
|       |               |___ images
|       |               |       |___ train
|       |               |       |       |___ frame_xxxxxx.jpg
|       |               |       |       |___ ...
|       |               |       |___ val
|       |               |               |___ frame_xxxxxx.jpg
|       |               |               |___ ...       
|       |               |        
|       |               |___ labels
|       |                       |___ train
|       |                       |       |___ frame_xxxxxx.txt
|       |                       |       |___ ...
|       |                       |___ val
|       |                               |___ frame_xxxxxx.txt
|       |                               |___ ...  
|	|
|	|__reid_dataset
|			|
|			|__query
|			|    |
|			|    |__id_1
|                       |    |    |__frame_xxxxxx.jpg
|			|    |
|			|    |__id_2
|                       |         |__frame_xxxxxx.jpg
|			|
|			|__gallery
|			|    |
|			|    |__id_1
|                       |    |    |__frame_xxxxxx.jpg
|                       |    |    |__...
|			|    |
|			|    |__id_2
|                       |         |__frame_xxxxxx.jpg
|                       |         |__...
|			|
|			|__train
|			     |
|			     |__id_1
|                            |    |__frame_xxxxxx.jpg
|                            |    |__...
|			     |
|			     |__id_2
|                                 |__frame_xxxxxx.jpg
|                                 |__...
|
|___data_version_n
|
|
|___combine_dataset
	 |
         |__TRAIN_DATASET
	 |
	 |__detection_dataset
	 |
         |__reid_dataset	

```
Where each `data_version` folder refer to a sub dataset. In each `data_version` folder, there are `TRAIN_DATASET` refer to general data tree, `detection_dataset` is formatted follow YOLOV5 input format, and `reid_dataset` follow Market1501 format.

There are also `combine_dataset`, which includes mix data of all previous versions.

Each data version will be updated into the designed tree as above.

### Generate Detection Data
Will need first prepare a frame folder and labels folder as below:

```bash
frames_folder_root
	|
	|___train
	|	| 
	|	|___ video_name_1
	|	|       |___ frame_xxxxxx.jpg
	|	|       |___ ...
	|	|___ video_name_2
	|		|___ frame_xxxxxx.jpg
	|		|___ ...
	|		
	|___test
		| 
		|___ video_name_1
		|       |___ frame_xxxxxx.jpg
		|       |___ ...
		|___ video_name_2
			|___ frame_xxxxxx.jpg
			|___ ...
```

```bash
labels_folder_root
	|
	|___train
	|	| 
	|	|___ video_name_1
	|	|       |___ frame_xxxxxx.txt
	|	|       |___ ...
	|	|___ video_name_2
	|		|___ frame_xxxxxx.txt
	|		|___ ...
	|		
	|___test
		| 
		|___ video_name_1
		|       |___ frame_xxxxxx.txt
		|       |___ ...
		|___ video_name_2
			|___ frame_xxxxxx.txt
			|___ ...
```
Where each video labels folder is in YOLO label format.

To do this, do the following steps:

```bash
cd generate_data
python generate_data_tree
```
Where `data_root` is the folder contains raw VTX data. We use this folder to search for specific videos. `gt_root` is the root folder contains CVAT annotation files. And `tree_root` is the root of the tree data tree we want to create.

We then symlink from these folder to the `MAIN_DATA_TREE` for saving storage and generating time.

```bash
cd generate_data
```

In `combine_train_detection_dataset.py`, declare `root_frames_dir` and `root_labels_dir` as path to these 2 above folder. Declare `combine_frames_dir` and `combine_labels_dir` as path to `detection_dataset` folder in `MAIN_DATA_TREE`. Run:

```bash 
python combine_train_detection_dataset.py
```

### Generate ReID Data 
Follow tutorial from this [repo](https://github.com/LeDuySon/ALL_SCRIPTS)

## Detection Module

### Training Custom Data
**Follow** [Training Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

**Note:**

0. Create data folder in the right format (YOLOV5 will check label paths corresponding to image paths.
1. In `data.yaml` file, set `nc = 1`, `name = [person]`. Replace `train` and `val` with absolute paths instead of relative paths as in the above tutorial.
2. `config yaml` file should be placed in `yolov5/data`.
3. Training with `crowdhuman_yolov5 checkpoint` need to set `Optimizer: ...` as `None` first, or else it'll be conflict during training.
4. File [hyp.scratch.yaml](https://github.com/ultralytics/yolov5/issues/607) in case it's not included in original repo (để trong folder yolov5/data/)
5. Specific GPU for training
6. If evaluating on different dataset using pretrained model, we need to remove `best_fitness` score of the checkpoint. Note line 155 in `yolov5/train.py` to remove `best_fitness` score of the checkpoint.
7. Training script

```bash
python train.py --data {data_yaml_file_config} --epochs {num_epochs} --batch {batches} --weights {weights path} --cfg {model config path} --device 0
```
If we use `crowdhuman_yolov5 checkpoint`, then we can use `yolov5m config file` in `yolov5/models/yolov5m.yaml`

Training result will be saved in `/yolov5/runs/train/exp{x}`. 

**Model best checkpoint after finetuned num class heads and training for 30 epoch on VTX DATA: [Checkpoint](https://wandb.ai/hainguyen/YOLOv5/artifacts/model/run_3gqwg2vr_model/ebe1245d78646d98df91/files)**

### Evaluate Detection Module (YOLOV5)
```bash
python test.py --data {data_yaml_file_config} --weights {weights_path} --save-txt --save-conf
```
Where `data config yaml` file set `train path` and `val path` as absolute path to images folder of test data (the model will test all images in the folder)

File label after evaluated will be save in `/yolov5/runs/test/exp{x}`

Evaluate result of `Finetune Model` on `VTX DATA` after training for 30 epochs and `Model pretrained on CrowdHman Dataset`: [Evaluation Results](https://docs.google.com/spreadsheets/d/1BOKNfHO-Ar7BzfpYRyFjux44B-I44XICpk7thhqd3MY/edit?fbclid=IwAR1GgUpXwZGpfFvW5TSdUTRWC09U4OIxLK2ajcDB218c0WngXt9ypyqVNhc#gid=0)


### Inferrence Detection Module (YOLOV5)
```bash
python detect.py --source {data_source_path} --weights {weights_path} --save-txt --save-conf
```
Where source can be path to 1 image or a whole image folder

Inference result is saved in `yolov5/runs/detect/exp{x}`

## ReID module
### Prepare data format for ReID module
The data format for ReID module is:
```bash
root
| 
|___ train
|       |___ id_1
|       |     |___ frame_xxxxxx.jpg
|       |     |___ ...
|       |
|       |___ id_2
|       |     |___ frame_xxxxxx.jpg
|       |     |___ ...       
|       |
|       |___ id_n
|             |___ frame_xxxxxx.jpg
|             |___ ...       
|        
|___ test
|       |___ id_1
|       |     |___ frame_xxxxxx.jpg
|       |     |___ ...
|       |
|       |___ id_2
|       |     |___ frame_xxxxxx.jpg
|       |     |___ ...       
|       |
|       |___ id_n
|             |___ frame_xxxxxx.jpg
|             |___ ...
|
|___ gallery
|       |___ id_1
|       |     |___ frame_xxxxxx.jpg
|       |     |___ ...
|       |
|       |___ id_2
|       |     |___ frame_xxxxxx.jpg
|       |     |___ ...       
|       |
|       |___ id_n
|             |___ frame_xxxxxx.jpg
|             |___ ...  
|
|___ query
        |___ id_1
        |     |___ frame_xxxxxx.jpg
        |     |___ ...
        |
        |___ id_2
        |     |___ frame_xxxxxx.jpg
        |     |___ ...       
        |
        |___ id_n
              |___ frame_xxxxxx.jpg
              |___ ...  
```

For `VTX Data`, the `train` folder contains 90% of the combine train data, the `test` folder contains the rest 10%. 

The `gallery` folder contains the whole test dataset. And the `query` folder is random splitted from `gallery`, 1 image for each id. 

### Training ReID module
The model have 2 checkpoints in the beginning. We can find those [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).

To use `ckpt.t7` weight, in `deep_sort_pytorch/deep_sort/deep/model.py`, use the `Author's finetuned model` and set `num_classes = 751`

To use `original_ckpt.t7` weight, in `deep_sort_pytorch/deep_sort/deep/model.py`, use the `Original model` and set `num_classes = 625`

To use the trained weight on `VTX DATA`, in `deep_sort_pytorch/deep_sort/deep/model.py`, use the `Author's finetuned model` and set `num_classes = 868`

Training script:

```bash
python train.py --data-dir {path/to/data/root/dir} --ckpt {path/to/pretrained/reid/checkpoint} --save-ckpt-path {path/to/save/best/checkpoint} --save-result {path/to/save/training/curve/image}
```

You can find more arguments in `deep_sort_pytorch/deep_sort/deep/train.py`

### Testing ReID module
This step is used to create a features matrix for evaluating result.

In `deep_sort_pytorch/deep_sort/deep`, run:

```bash
python test.py --data-dir {path/to/data/root/dir} --ckpt {path/to/reid/checkpoint} --save-path {path/to/save/features/metric}
```

### Evaluating ReID module
These below functions take a dictionary `features` as input. The `features` dictionary includes the following keys:
```bash
qf: matrix of vector features for each query
ql: matrix of query labels
gf: matrix of vector features for each gallery
gl: matrix of gallery labels
query_paths: list of paths for all query images 	
gallery_paths: list of paths for all gallery images	
```

**1. Evaluating on the whole gallery for all queries**

In `deep_sort_pytorch/deep_sort/deep`, run:

```bash
python evaluate.py --predict-path {path/to/saved/features/metric} --p_k {k in P@k evaluation} --mAP_n {n in mAP@n evaluation}
```

**2. Evaluating each query on a gallery base on frame id of each query** 

Since the ReID module in Deepsort mainly focus on solving the ID switch problem in tracking process, it's unnecessary to search a query on the whole gallery. Instead of that, we just need to evaluate a query in a certain frame length. 

For example, a query instance that appear in frame `x` just need to be evaluated on a gallery with all instance from frame `x - range` to frame `x + range`, where `range` is a pre-defined number (we set range = 100 by default).

In `deep_sort_pytorch/deep_sort/deep`, run:

```bash
python evaluate_frame_base.py --predict-path {path/to/saved/features/metric} --p_k {k in P@k evaluation} --mAP_n {n in mAP@n evaluation}
```

**3. Evaluating each query on a gallery base on trajectory of each person id** 
In this algorithm, for each person id, we first find the first frame the id appears and the last frame the id appears, and define them as `start_trajectory_frame` and `end_trajectory_frame` of the id's trajectory. 

We then evaluate a query on a gallery with all instance from frame `start_trajectory_frame - range` to frame `end_trajectory_frame + range`, where `range` is a pre-defined number (we set range = 100 by default).

However, this algorithm is almost similar as the 2th algorithm (base on query frame id), but run quite slower.

In `deep_sort_pytorch/deep_sort/deep`, run:

```bash
python evaluate_trajectory_base.py --predict-path {path/to/saved/features/metric} --p_k {k in P@k evaluation} --mAP_n {n in mAP@n evaluation}
```

Note that since each gallery is only in a limited number of frame, there can be some cases that the number of instances is smaller than `k` and `n` in `p@k` and `mAP@n` evaluation. However, it's quite rare because `k` and `n` is small (5 or 10). We can solve it easily by just ignore that gallery and reference query.

There is also optional param for showing top k matched images for each query.

## Tracking

### Additional tracking source
**Tracking with video (default):**

```bash
python track_video.py --source {path/to/mp4/video} 
```

**Tracking with ensemble predicted result instead:**

```bash
python track.py --frame_dir {path/to/frame/dir} --det_pred_dir {path/to/ensemble/predict/dir} --gt_path {path/to/gt/file} --output {path/to/output/dir} --save-txt 
```
Where `det_pred_dir` is in `mmdetection` predict format, which is `<class_name> <confidence> <left> <top> <right> <bottom>` for each `txt` file. `gt file` is in `MOT` format.

The `output` is in `MOT` format, which is `<frame>, <id>, <bb_top>, <bb_left>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>`.

### Tracking Evaluation
Following tutorial from [this repo](https://github.com/ConstantSun/MOT_Evaluation)

# Yolov5 + Deep Sort with PyTorch





<div align="center">
<p>
<img src="MOT16_eval/track_pedestrians.gif" width="400"/> <img src="MOT16_eval/track_all.gif" width="400"/> 
</p>
<br>
<div>
<a href="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/actions"><img src="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<br>  
<a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 
</div>

</div>


## Introduction

This repository contains a two-stage-tracker. The detections generated by [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset, are passed to a [Deep Sort algorithm](https://github.com/ZQPei/deep_sort_pytorch) which tracks the objects. It can track any object that your Yolov5 model was trained to detect.


## Tutorials

* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [Deep Sort deep descriptor training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;
* [Yolov5 deep_sort pytorch evaluation](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation)&nbsp;



## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats

```bash
python3 track.py --source ... --show-vid  # show live inference results as well
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`


## Select a Yolov5 family model

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov5 family model for automatic download

```bash
python3 track.py --source 0 --yolo_weights yolov5s.pt --img 640  # smallest yolov5 family model
```

```bash
python3 track.py --source 0 --yolo_weights yolov5x6.pt --img 1280  # largest yolov5 family model
```


## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you only want to track persons I recommend you to get [these weights](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) for increased performance

```bash
python3 track.py --source 0 --yolo_weights yolov5/weights/crowdhuman_yolov5m.pt --classes 0  # tracks persons, only
```

If you want to track a subset of the MS COCO classes, add their corresponding index after the classes flag

```bash
python3 track.py --source 0 --yolo_weights yolov5s.pt --classes 16 17  # tracks cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov5 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.


## MOT compliant results

Can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```


## Cite

If you find this project useful in your research, please consider cite:

```latex
@misc{yolov5deepsort2020,
    title={Real-time multi-object tracker using YOLOv5 and deep sort},
    author={Mikel Broström},
    howpublished = {\url{https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch}},
    year={2020}
}
```


## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.


# draw pred and gt boxes:
 File track.py:
- Comment line 298
- Remove opt.mode in line 216, 234

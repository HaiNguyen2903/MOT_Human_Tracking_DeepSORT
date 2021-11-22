# convert label_with_ids to yolov5 format

import os
from os import path as osp

src_root = '/data/DATA_ROOT/combine_dataset_v5/UET_MOT'

src_label_dir = osp.join(src_root, 'labels_with_ids')
src_frame_dir = osp.join(src_root, 'images')

# yolov5_label_dir = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset_v3/detection_dataset/labels'
# yolov5_frame_dir = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset_v3/detection_dataset/images'

yolov5_label_dir = '/data/DATA_ROOT/combine_dataset_v5/detection_dataset/labels'
yolov5_frame_dir = '/data/DATA_ROOT/combine_dataset_v5/detection_dataset/images'


src_label_train = osp.join(src_label_dir, 'train')
src_label_test = osp.join(src_label_dir, 'test')
src_label_val = osp.join(src_label_dir, 'val')

dest_label_train = osp.join(yolov5_label_dir, 'train')
dest_label_test = osp.join(yolov5_label_dir, 'test')
dest_label_val = osp.join(yolov5_label_dir, 'val')


def mkdir_if_missing(path):
    if not os.path.exists(path):
        print('Make dir {}'.format(path))
        os.makedirs(path)


# format train labels

for vid in sorted(os.listdir(src_label_train)):
    dest_gt_dir = osp.join(dest_label_train, vid)
    mkdir_if_missing(dest_gt_dir)

    gt_dir = osp.join(src_label_train, vid, 'img1')

    for gt in sorted(os.listdir(gt_dir)):
        print('Writing {} for video {}'.format(gt[:-4], vid))
        with open(osp.join(gt_dir, gt), 'r') as src_gt:

            with open(osp.join(dest_gt_dir, gt), 'w') as dest_gt:
                dest_gt.close()

            src_data = src_gt.readlines()
            for line in src_data:
                line = line.strip()
                ls = line.split(" ")
                
                with open(osp.join(dest_gt_dir, gt), 'a') as dest_gt:
                    dest_gt.write("{} {} {} {} {}\n".format(ls[0], ls[2], ls[3], ls[4], ls[5]))
    print()


print('Generate format for test videos\n')
    

# format test labels

for vid in sorted(os.listdir(src_label_test)):
    dest_gt_dir = osp.join(dest_label_test, vid)
    mkdir_if_missing(dest_gt_dir)

    gt_dir = osp.join(src_label_test, vid, 'img1')

    for gt in sorted(os.listdir(gt_dir)):
        print('Writing {} for video {}'.format(gt[:-4], vid))
        with open(osp.join(gt_dir, gt), 'r') as src_gt:
            with open(osp.join(dest_gt_dir, gt), 'w') as dest_gt:
                dest_gt.close()

            src_data = src_gt.readlines()
            for line in src_data:
                line = line.strip()
                ls = line.split(" ")
                
                with open(osp.join(dest_gt_dir, gt), 'a') as dest_gt:
                    dest_gt.write("{} {} {} {} {}\n".format(ls[0], ls[2], ls[3], ls[4], ls[5]))
    print()


print('Generate format for val videos\n')
    

# format test labels

for vid in sorted(os.listdir(src_label_val)):
    dest_gt_dir = osp.join(dest_label_val, vid)
    mkdir_if_missing(dest_gt_dir)

    gt_dir = osp.join(src_label_val, vid, 'img1')

    for gt in sorted(os.listdir(gt_dir)):
        print('Writing {} for video {}'.format(gt[:-4], vid))
        with open(osp.join(gt_dir, gt), 'r') as src_gt:
            with open(osp.join(dest_gt_dir, gt), 'w') as dest_gt:
                dest_gt.close()

            src_data = src_gt.readlines()
            for line in src_data:
                line = line.strip()
                ls = line.split(" ")
                
                with open(osp.join(dest_gt_dir, gt), 'a') as dest_gt:
                    dest_gt.write("{} {} {} {} {}\n".format(ls[0], ls[2], ls[3], ls[4], ls[5]))
    print()


def cal_len(path):
    return len(os.listdir(path))

assert cal_len(src_label_train) == cal_len(dest_label_train)
assert cal_len(src_label_test) == cal_len(dest_label_test)


for vid in os.listdir(dest_label_train):
    assert cal_len(osp.join(dest_label_train, vid)) == cal_len(osp.join(src_label_train, vid, 'img1'))

for vid in os.listdir(dest_label_test):
    assert cal_len(osp.join(dest_label_test, vid)) == cal_len(osp.join(src_label_test, vid, 'img1'))

print('Copy successfully')
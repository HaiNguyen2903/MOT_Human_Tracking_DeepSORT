import os
from os import path as osp

src_frame = '/data/DATA_ROOT/combine_dataset/detection_dataset/images/train'
src_label = '/data/DATA_ROOT/combine_dataset/detection_dataset/labels/train'

dest_frame = '/data/hain/toy_data/toy_detection_dataset/images/train'
dest_label = '/data/hain/toy_data/toy_detection_dataset/labels/train'

num = 50

def mkdir_if_missing(full_path):
    if not osp.exists(full_path):
        print('Create dir {}'.format(full_path))
        os.makedirs(full_path)

mkdir_if_missing(dest_frame)
mkdir_if_missing(dest_frame)

for img in sorted(os.listdir(src_frame)):
    id = int(img[6:12])
    for i in range(num):
        if not os.path.exists(os.path.join(dest_frame, 'frame_{:06d}.jpg'.format(id + i))):
            os.symlink(os.path.join(src_frame, img), os.path.join(dest_frame, 'frame_{:06d}.jpg'.format(id + i)))


for gt in sorted(os.listdir(src_label)):
    id = int(gt[6:12])
    for i in range(num):
        if not os.path.exists(os.path.join(dest_label, 'frame_{:06d}.jpg'.format(id + i))):
            os.symlink(os.path.join(src_label, gt), os.path.join(dest_label, 'frame_{:06d}.txt'.format(id + i)))



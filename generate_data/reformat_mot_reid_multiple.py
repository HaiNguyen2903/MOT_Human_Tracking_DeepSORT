# Reformat MOT20 dataset to ReID dataset for DeepSORT ReID module

import os 
from os import path as osp
import cv2
import glob
import shutil

root = '/data/MOT20/MOT20/train'

# new reid tree generate from MOT dataset (for single video in MOT dataset first)

reid_root = '/data/MOT20/MOT20/MOT20_REID_FORMAT'

train = osp.join(reid_root, 'train')
query = osp.join(reid_root, 'query')
gallery = osp.join(reid_root, 'gallery')

def mkdir_if_missing(path):
    if not os.path.exists(path):
        print('Make dir {}'.format(path))
        os.makedirs(path) 

def crop_image(src_path, x, y, w, h, save_path):
    img = cv2.imread(src_path)
    cropped_img = img[y:y+h, x:x+w]
    print('crop image to {}'.format(save_path))
    cv2.imwrite(save_path, cropped_img)


def split_gallery_query(query_path, gallery_path, query_sample=1):
    track_folder = os.listdir(gallery_path)
    
    for t in track_folder:
        save_folder = os.path.join(query_path, t)
        target_folder = os.path.join(gallery_path, t)
        
        imgs = glob.glob(target_folder+"/*.jpg")

        mkdir_if_missing(save_folder)
       # print(imgs)
        if(len(imgs) == 1):
            continue
        #num_test = min(test_min, int(len(imgs) * ratio))
        #num_test = max(num_test, 1)
        import random
        query_img_ls = random.sample(imgs, query_sample)
        #print(test_img_ls)
        for img in query_img_ls:
            shutil.move(img, save_folder)


mkdir_if_missing(reid_root)
mkdir_if_missing(train)
mkdir_if_missing(query)
mkdir_if_missing(gallery)


for subdir in os.listdir(root):
    num_ids = 0

    gt_path = osp.join(root, subdir, 'gt/gt.txt')
    img_dir = osp.join(root, subdir, 'img1')


    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(',')
            frame_idx, pid, x, y, w, h, _, _, _ = line

            src_img = osp.join(img_dir, '{:06d}.jpg'.format(int(frame_idx)))

            # print(src_img)
            # exit()
            save_path = osp.join(gallery, str(int(pid) + num_ids))
            
            mkdir_if_missing(save_path)

            crop_image(src_path=src_img, x=int(x), y=int(y), w=int(w), h=int(h), save_path=osp.join(save_path, '{}.jpg'.format(int(frame_idx))))

    num_ids += len(os.listdir(gallery))


split_gallery_query(query, gallery, query_sample=1)

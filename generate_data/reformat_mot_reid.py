# Reformat MOT20 dataset to ReID dataset for DeepSORT ReID module

import os 
from os import path as osp
import cv2
import glob
import shutil

root = '/data/MOT15/MOT15/train/PETS09-S2L1'

# new reid tree generate from MOT dataset (for single video in MOT dataset first)

reid_root = '/data/MOT15/MOT15/PETS09-S2L1-REID-FORMAT'

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


gt_path = osp.join(root, 'gt/gt.txt')
img_dir = osp.join(root, 'img1')

with open(gt_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(',')
        frame_idx, pid, x, y, w, h, _, _, _, _ = line 

        frame_idx = int(frame_idx)
        x, y, w, h = float(x), float(y), float(w), float(h)

        src_img = osp.join(img_dir, '{:06d}.jpg'.format(int(frame_idx)))

        # print(src_img)
        # exit()
        save_path = osp.join(gallery, pid)
        
        mkdir_if_missing(save_path)

        crop_image(src_path=src_img, x=int(x), y=int(y), w=int(w), h=int(h), save_path=osp.join(save_path, '{}.jpg'.format(int(frame_idx))))

split_gallery_query(query, gallery, query_sample=1)

# for id in os.listdir(gallery):
#     for img in os.listdir(os.path.join(gallery, id)):
#         # frame_idx = str(int(img[:-4]))
#         # os.rename(os.path.join(gallery, id, img), os.path.join(gallery, id, frame_idx + '.jpg'))

#         if 'frame' in img:
#             frame_idx = str(int(img[:-4].split('_')[1]))
#             os.rename(os.path.join(gallery, id, img), os.path.join(gallery, id, frame_idx + '.jpg'))

# for id in os.listdir(query):
#     for img in os.listdir(os.path.join(query, id)):
#         frame_idx = str(int(img[:-4]))
#         os.rename(os.path.join(query, id, img), os.path.join(query, id, frame_idx + '.jpg'))
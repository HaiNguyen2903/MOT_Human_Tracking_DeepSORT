import cv2
import math
import numpy as np
import argparse
import os
from gt_utils import get_object_frame
import glob
import shutil
# Make sure that the print function works on Python 2 and 3
# Capture every n seconds (here, n = 5) 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_path', '-V', type=str,
                    help='video path to extract frame')
parser.add_argument('--save_path', '-sp', type=str, 
                    help="save frame video path")
args = parser.parse_args()

#################### Setting up the file ################
videoFile = args.video_path
print(videoFile)
vidcap = cv2.VideoCapture(videoFile)
success, image = vidcap.read()
if(success):
    print("Capture video successfully")
else:
    print("Failed")


def create_folder(path):
    if not os.path.exists(path):
        print("Create folder: ", path)
        os.mkdir(path)
def check_num_files(path):
    if os.path.exists(path):
        return len(os.listdir(path))
    return 0

train_path = "reid_dataset/train"
test_path = "reid_dataset/test"

create_folder(train_path)
create_folder(test_path)
#################### Setting up parameters ################
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print("Frame rate: ", fps)

limit_train = 50
group_frame = get_object_frame("gt.txt")
################### Initiate Process ################
while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    if(frameId % 1000 == 0):
        print("Process frame: ", frameId)
    success, image = vidcap.read()
    if(image is None):
        continue
    if(len(group_frame[frameId]) != 0):
        for idx, obj in enumerate(group_frame[frameId]):
            x, y, w, h = list(map(int, obj.coord))
            print("Coord: ", obj.coord)
            save_path = os.path.join(train_path, f"{obj.track_id}")
           # if(check_num_files(save_path) > limit_train):
           #     save_path = os.path.join(test_path, f"{obj.track_id}")
            crop_obj = image[y:y+h, x:x+w]
            create_folder(save_path)
            save_crop_name = os.path.join(save_path, "{}.jpg".format(frameId))
            print(save_crop_name)
            cv2.imwrite(save_crop_name, crop_obj)
       # if(frameId == 10):
       #     break
            


        
vidcap.release()
print("Complete")

def split_train_test(ratio = 0.2, test_min = 5):
    track_folder = os.listdir(train_path)
    for t in track_folder:
        save_folder = os.path.join(test_path, t)
        target_folder = os.path.join(train_path, t)

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        imgs = glob.glob(target_folder+"/*.jpg")
       # print(imgs)
        if(len(imgs) == 1):
            continue
        num_test = min(test_min, int(len(imgs) * ratio))
        num_test = max(num_test, 1)
        import random
        test_img_ls = random.sample(imgs, num_test)
        #print(test_img_ls)
        for img in test_img_ls:
            shutil.move(img, save_folder)
print("Split train test")
split_train_test()


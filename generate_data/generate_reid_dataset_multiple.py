import cv2
import math
import numpy as np
import argparse
import os
from gt_utils import get_object_frame
import glob
import shutil
# Make sure that the print function works on Python 2 and 3

#################### Setting up the file ################i

def create_folder(path):
    if not os.path.exists(path):
        #print("Create folder: ", path)
        os.mkdir(path)

def check_num_files(path):
    if os.path.exists(path):
        return len(os.listdir(path))
    return 0

# set up save path

def init_path(args):
    save_path = args.save_path
    train_path = os.path.join(save_path, "train")
    gallery_path = os.path.join(save_path, "gallery")
    query_path = os.path.join(save_path, "query")
    
    paths = [train_path, gallery_path, query_path]
    
    create_folder(save_path)
    for path in paths:
        create_folder(path)

    return paths    

def generate_frames(video_path, train_path):
    global CUR_NUM_ID
    print(CUR_NUM_ID)
    videoFile = video_path
    print(videoFile)
    vidcap = cv2.VideoCapture(videoFile)
    success, image = vidcap.read()
    if(success):
        print("Capture video successfully")
    else:
        print("Failed")
    
    #################### Setting up parameters ################
    fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
    print("Frame rate: ", fps)
    # -4 for exclude .mp4, just get dataset name
    gt_path = video_path[:-4] + "/gt/gt.txt"
    print("GT path: ", gt_path)
    group_frame = get_object_frame(gt_path)
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
                #print("Coord: ", obj.coord)
                obj_id = obj.track_id + CUR_NUM_ID
                save_path = os.path.join(train_path, f"{obj_id}")
               # if(check_num_files(save_path) > limit_train):
               #     save_path = os.path.join(test_path, f"{obj.track_id}")
                crop_obj = image[y:y+h, x:x+w]
                # resize all images to fixed size
                crop_obj = cv2.resize(crop_obj, IMG_SIZE)
                create_folder(save_path)
                save_crop_name = os.path.join(save_path, "{}.jpg".format(frameId))
                #print(save_crop_name)
                cv2.imwrite(save_crop_name, crop_obj)
    # keep track number of ids each video
    CUR_NUM_ID = len(os.listdir(train_path)) -1 
    
    vidcap.release()
    print("Complete: ", video_path)

def split_train_gallery(train_path, gallery_path, ratio = 0.3):
    """ratio(float): split ratio between train dataset and gallery dataset
        (default=1) just for test reid model"""

    track_folder = os.listdir(train_path)
    num_id_gallery = int(len(track_folder) * ratio)
    num_ids = len(track_folder)
    
    import random 
    random.shuffle(track_folder)

    
    for t in track_folder:
        if(num_id_gallery == 0):
            break
        if((int(t) % 2 == 0 and num_ids % 2 == 0 and ratio != 1) or ratio == 0):
            continue
        save_folder = os.path.join(gallery_path, t)
        target_folder = os.path.join(train_path, t)
        shutil.move(target_folder, save_folder)
    
        #imgs = glob.glob(target_folder+"/*.jpg")
       ## print(imgs)
#        if(len(imgs) == 1):
#            continue
#        num_test = min(5, int(len(imgs) * ratio))
#        num_test = max(num_test, 1)
#        import random
#        test_img_ls = random.sample(imgs, num_test)
#        ##print(test_img_ls)
#        for img in test_img_ls:
#            shutil.move(img, save_folder)
#
        num_id_gallery -= 1

def split_gallery_query(query_path, gallery_path, query_sample=1):
    track_folder = os.listdir(gallery_path)
    
    for t in track_folder:
        save_folder = os.path.join(query_path, t)
        target_folder = os.path.join(gallery_path, t)
        
        imgs = glob.glob(target_folder+"/*.jpg")

        create_folder(save_folder)
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

def main(args):
    # create dataset folder 
    train_path, gallery_path, query_path = init_path(args)

    # get all video in data path
    videos = glob.glob(args.data_path + "/*.mp4")
    # generate frame for each video
    for video in videos:
        generate_frames(video, train_path)
    
    print("Split train gallery query")
    split_train_gallery(train_path, gallery_path, args.gallery_ratio)
    split_gallery_query(query_path, gallery_path, args.num_query)

if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', '-dp', type=str,
                        help='video path to extract frame')
    parser.add_argument('--save_path', '-sp', type=str, 
                        help="save frame video path")
    parser.add_argument('--mode', '-m', type=str,
                        help="split data mode train|test|all")
    parser.add_argument('--gallery_ratio', '-gr', type=float, default=0.3,
                        help="train and gallery ratio split")
    parser.add_argument('--num_query', '-nq', type=int,
                        default=1, help="number query per id")
    args = parser.parse_args()
    
    if(args.mode == "train"):
        print("TRAIN")
        args.gallery_ratio = 0
        args.num_query = 0
    elif(args.mode == "test"):
        print("TEST")
        args.gallery_ratio = 1
        args.num_query = 1
    
    # Config params
    IMG_SIZE = (256, 128) # h, w
    CUR_NUM_ID = -1 # number of current ids, -1 because our dataset is 0 based but MOT format is 1 based

    print("START")
    main(args)
import os 

'''
This file combine frames data from multi folders to generate 1 unique training dataset
'''

root_frames_dir = '/data.local/hangd/data_vtx/frames_data/train'
root_labels_dir = '/data.local/hangd/data_vtx/detection_dataset/labels/train_gt'

combine_frames_dir = '../combine_dataset/images/train'
combine_labels_dir = '../combine_dataset/labels/train'

assert os.path.isdir(root_frames_dir)
assert os.path.isdir(root_labels_dir)

assert os.path.isdir(combine_frames_dir)
assert os.path.isdir(combine_labels_dir)

train_vid_order = [
    'NVR-CH01_S20210607-094253_E20210607-094856',
    'NVR-CH02_S20210607-112840_E20210607-120113',
    'NVR-CH01_S20210607-095007_E20210607-095126',
    'NVR-CH07_S20210609-112939_E20210609-113836',
    'NVR-CH01_S20210607-113251_E20210607-120452'
]

current_frame = 0

frame_exception_files = []
label_exception_files = []
'''
Unnote to create symlink
'''
for i in range(len(train_vid_order)):
    length = len(os.listdir(os.path.join(root_frames_dir, train_vid_order[i])))

    for img in os.listdir(os.path.join(root_frames_dir, train_vid_order[i])):
        id = int(img[6:12]) + current_frame
        os.symlink(os.path.join(root_frames_dir, train_vid_order[i], img), os.path.join(combine_frames_dir, 'frame_{:06d}.jpg'.format(id)))
        

    for img in os.listdir(os.path.join(root_labels_dir, train_vid_order[i])):
        id = int(img[6:12]) + current_frame
        os.symlink(os.path.join(root_labels_dir, train_vid_order[i], img), os.path.join(combine_labels_dir, 'frame_{:06d}.txt'.format(id)))
    current_frame += length+1

total_frame = 0

for i in range(len(train_vid_order)):
    length = len(os.listdir(os.path.join(root_frames_dir, train_vid_order[i])))
    total_frame += length

print('Total frames:', total_frame)

assert len(os.listdir(combine_frames_dir)) == len(os.listdir(combine_labels_dir)) == total_frame

print('Successed.')
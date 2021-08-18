import os 
from os import path as osp
import shutil

'''
This file combine frames data from multi folders to generate 1 unique training dataset
'''

root_frames_dir = '/data.local/hangd/data_vtx/frames_data'
root_labels_dir = '/data.local/hangd/data_vtx/labels_yolo_format'


combine_frames_dir = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/detection_dataset/images'
combine_labels_dir = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/detection_dataset/labels'


current_frame = 0 

frame_exception_files = []
label_exception_files = []

def mkdir_if_missing(full_path):
    if not osp.exists(full_path):
        print('Create dir {}'.format(full_path))
        os.makedirs(full_path)

mkdir_if_missing(combine_frames_dir)
mkdir_if_missing(combine_labels_dir)


def handle_missing_files(frame_dir, label_dir):
    frame_len = len(os.listdir(frame_dir))
    label_len = len(os.listdir(label_dir))

    if frame_len == label_len:
        return

    elif frame_len < label_len:
        curr_last = 'frame_{:06d}.jpg'.format(frame_len - 1)

        if not os.path.exists(os.path.join(frame_dir, curr_last)):
            curr_last = curr_last[:-3] + 'PNG'

        print('Adding 1 more frame for video {}'.format(os.path.basename(frame_dir)))
        shutil.copyfile(os.path.join(frame_dir, curr_last), os.path.join(frame_dir, 'frame_{:06d}.jpg'.format(frame_len)))
        
    else:
        curr_last = 'frame_{:06d}.jpg'.format(frame_len - 1)

        if not os.path.exists(os.path.join(frame_dir, curr_last)):
            curr_last = curr_last[:-3] + 'PNG'

        print('Removing 1 last frame for video {}'.format(os.path.basename(frame_dir)))
        os.remove(os.path.join(frame_dir, current_frame))



def combine_symlink(root_frames_dir, root_labels_dir, combine_frames_dir, combine_labels_dir):
    current_frame = 0

    train_names = [name for name in sorted(os.listdir(os.path.join(root_frames_dir, 'train')))]
    test_names = [name for name in sorted(os.listdir(os.path.join(root_frames_dir, 'test')))]

    # print(train_names)
    # print()
    # print(test_names)

    vid_idx = 1
    
    for name in train_names:

        frames_src_dir = os.path.join(root_frames_dir, 'train', name)
        labels_src_dir = os.path.join(root_labels_dir, 'train', name)

        frames_dest_dir = os.path.join(combine_frames_dir, 'train')
        labels_dest_dir = os.path.join(combine_labels_dir, 'train')

        mkdir_if_missing(frames_dest_dir)
        mkdir_if_missing(labels_dest_dir)

        frame_len = len(os.listdir(frames_src_dir))
        label_len = len(os.listdir(labels_src_dir))

        handle_missing_files(frames_src_dir, labels_src_dir)

        assert frame_len == label_len, "Frame len: {} \t Label len: {}".format(frame_len, label_len)

        for frame in sorted(os.listdir(frames_src_dir)):
            id = int(frame[6:12]) + current_frame

            try:
                os.symlink(os.path.join(frames_src_dir, frame), os.path.join(frames_dest_dir, 'frame_{:06d}.jpg'.format(id)))
                os.symlink(os.path.join(labels_src_dir, frame[:-3] + 'txt'), os.path.join(labels_dest_dir, 'frame_{:06d}.jpg'.format(id)))
                print('Generate frame {} for train video number {}'.format(id, vid_idx))
            except:
                print('Already generated frame {} for test video number {}'.format(id, vid_idx))

        current_frame += frame_len+1
        vid_idx += 1

    assert len(os.listdir(frames_dest_dir)) == len(os.listdir(labels_dest_dir))

    # reset to 0
    current_frame = 0

    vid_idx = 1

    for name in test_names:

        frames_src_dir = os.path.join(root_frames_dir, 'test', name)
        labels_src_dir = os.path.join(root_labels_dir, 'test', name)

        frames_dest_dir = os.path.join(combine_frames_dir, 'test')
        labels_dest_dir = os.path.join(combine_labels_dir, 'test')

        mkdir_if_missing(frames_dest_dir)
        mkdir_if_missing(labels_dest_dir)

        frame_len = len(os.listdir(frames_src_dir))
        label_len = len(os.listdir(labels_src_dir))

        handle_missing_files(frames_src_dir, labels_src_dir)

        assert frame_len == label_len, "Frame len: {} \t Label len: {}".format(frame_len, label_len)

        for frame in sorted(os.listdir(frames_src_dir)):
            id = int(frame[6:12]) + current_frame
            try:
                os.symlink(os.path.join(frames_src_dir, frame), os.path.join(frames_dest_dir, 'frame_{:06d}.jpg'.format(id)))
                os.symlink(os.path.join(labels_src_dir, frame[:-3] + 'txt'), os.path.join(labels_dest_dir, 'frame_{:06d}.jpg'.format(id)))
                print('Generate frame {} for test video number {}'.format(id, vid_idx))
            except:
                print('Already generated frame {} for test video number {}'.format(id, vid_idx))

        current_frame += frame_len+1    
        vid_idx += 1

    assert len(os.listdir(frames_dest_dir)) == len(os.listdir(labels_dest_dir))

def cal_total_files(dir):
    count = 0
    for subdir in os.listdir(dir):
        count += len(os.listdir(os.path.join(dir, subdir)))

    return count

combine_symlink(root_frames_dir, root_labels_dir, combine_frames_dir, combine_labels_dir)

# assert os.path.isdir(root_frames_dir)
# assert os.path.isdir(root_labels_dir)

# assert os.path.isdir(combine_frames_dir)
# assert os.path.isdir(combine_labels_dir)

# train_vid_order = [
#     'NVR-CH01_S20210607-094253_E20210607-094856',
#     'NVR-CH02_S20210607-112840_E20210607-120113',
#     'NVR-CH01_S20210607-095007_E20210607-095126',
#     'NVR-CH07_S20210609-112939_E20210609-113836',
#     'NVR-CH01_S20210607-113251_E20210607-120452'
# ]

# current_frame = 0

# frame_exception_files = []
# label_exception_files = []
# '''
# Unnote to create symlink
# '''
# for i in range(len(train_vid_order)):
#     length = len(os.listdir(os.path.join(root_frames_dir, train_vid_order[i])))

#     for img in os.listdir(os.path.join(root_frames_dir, train_vid_order[i])):
#         id = int(img[6:12]) + current_frame
#         os.symlink(os.path.join(root_frames_dir, train_vid_order[i], img), os.path.join(combine_frames_dir, 'frame_{:06d}.jpg'.format(id)))
        

#     for img in os.listdir(os.path.join(root_labels_dir, train_vid_order[i])):
#         id = int(img[6:12]) + current_frame
#         os.symlink(os.path.join(root_labels_dir, train_vid_order[i], img), os.path.join(combine_labels_dir, 'frame_{:06d}.txt'.format(id)))
#     current_frame += length+1

# total_frame = 0

# for i in range(len(train_vid_order)):
#     length = len(os.listdir(os.path.join(root_frames_dir, train_vid_order[i])))
#     total_frame += length

# print('Total frames:', total_frame)

# assert len(os.listdir(combine_frames_dir)) == len(os.listdir(combine_labels_dir)) == total_frame


assert cal_total_files(os.path.join(root_frames_dir, 'train')) == len(os.listdir(os.path.join(combine_frames_dir, 'train')))
assert cal_total_files(os.path.join(root_frames_dir, 'test')) == len(os.listdir(os.path.join(combine_frames_dir, 'test')))

print('Successed.')

print('Total train frames:', len(os.listdir(os.path.join(combine_frames_dir, 'train'))))
print('Total test frames:', len(os.listdir(os.path.join(combine_frames_dir, 'test'))))
import os
import cv2
from IPython import embed

train_vids = ["NVR-CH02_S20210607-173836_E20210607-173936.mp4",
              "NVR-CH02_S20210608-083623_E20210608-083906.mp4",
              "NVR-CH02_S20210608-084321_E20210608-084631.mp4",
              "NVR-CH02_S20210608-085112_E20210608-085718.mp4",
              "NVR-CH02_S20210608-085752_E20210608-085950.mp4",
              "NVR-CH02_S20210608-114057_E20210608-114337.mp4",
              "NVR-CH02_S20210608-114346_E20210608-114508.mp4",
              "NVR-CH02_S20210608-114801_E20210608-115058.mp4",
              "NVR-CH02_S20210608-115057_E20210608-120139.mp4",
              "NVR-CH02_S20210609-083605_E20210609-083710.mp4",
              "NVR-CH02_S20210609-083729_E20210609-083949.mp4",
              "NVR-CH02_S20210609-085200_E20210609-085751.mp4",
              "NVR-CH02_S20210609-115107_E20210609-115831.mp4",
              "NVR-CH02_S20210609-173131_E20210609-173358.mp4",
              "NVR-CH02_S20210610-084425_E20210610-085713.mp4"]


test_vids = ["NVR-CH02_S20210607-094253_E20210607-094604.mp4",
             "NVR-CH02_S20210607-094604_E20210607-094856.mp4",
             "NVR-CH02_S20210607-170441_E20210607-170551.mp4",
             "NVR-CH02_S20210608-110218_E20210608-110328.mp4",
             "NVR-CH02_S20210608-120341_E20210608-120510.mp4",
             "NVR-CH02_S20210608-182322_E20210608-182524.mp4",
             "NVR-CH02_S20210609-172459_E20210609-172604.mp4",
             "NVR-CH02_S20210609-172604_E20210609-173139.mp4"
             ]

train_names = [vid[:-4] for vid in train_vids]
test_names = [vid[:-4] for vid in test_vids]

# root of videos data in generated data tree
data_root = '/data.local/hangd/data_vtx/data_17_08_21/DATA_TREE'

# gt root for all videos data
gt_root = '/data.local/hangd/data_vtx/data_17_08_21/labels/annotationCH02_reviewed'

# save frames dir 
frames_dir = '/data.local/hangd/data_vtx/frames_data'


def mkdir_if_missing(full_path):
    if not os.path.exists(full_path):
        print('Create dir {}'.format(full_path))
        os.makedirs(full_path)


def extract_frame(vid_path, save_folder, vid_idx, type):
    assert os.path.isfile(vid_path)
    assert os.path.isdir(save_folder)

    cap= cv2.VideoCapture(vid_path)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(save_folder, 'frame_{:06d}.jpg'.format(i)), frame)
        print('Extracting frame {} for {} video number {}'.format(i, type, vid_idx))
        i+=1

    print('Finish extracting {} video number {}'.format(type, vid_idx))
    print()
    cap.release()
    cv2.destroyAllWindows()


# generate root frame tree (cotinue generate from last dataset)
def gen_frame_data_tree(root_tree, data_root, train_names, test_names):
    mkdir_if_missing(root_tree)

    train_idx = 1

    test_idx = 1

    for name in train_names:

        full_path = os.path.join(root_tree, 'train', name)

        mkdir_if_missing(full_path)

        extract_frame(os.path.join(data_root, 'train', name, name + '.mp4'), os.path.join(full_path), train_idx, 'train')

        train_idx += 1

    for name in test_names:
        full_path = os.path.join(root_tree, 'test', name)

        mkdir_if_missing(full_path)

        extract_frame(os.path.join(data_root, 'test', name, name + '.mp4'), os.path.join(full_path), test_idx, 'test')

        test_idx += 1



gen_frame_data_tree(root_tree=frames_dir, data_root=data_root, train_names=train_names, test_names=test_names)
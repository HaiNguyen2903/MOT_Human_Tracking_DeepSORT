import os
import shutil

val_vids = [
    'NVR-CH01_S20210607-102303_E20210607-102433',
    'NVR-CH02_S20210608-114346_E20210608-114508',
    'NVR-CH02_S20210608-182322_E20210608-182524',
    'NVR-CH02_S20210609-115107_E20210609-115831',
    'NVR-CH04_S20210609-173203_E20210609-175447',
    'NVR-CH06_S20210607-131815_E20210607-131952',
    'NVR-CH06_S20210608-111753_E20210608-111910',
    'NVR-CH06_S20210608-112537_E20210608-112642',
    'NVR-CH06_S20210609-111136_E20210609-111336',
    'NVR-CH06_S20210609-111615_E20210609-111734'
]

test_vids = [
    'NVR-CH01_S20210607-095126_E20210607-102303',
    'NVR-CH02_S20210607-094604_E20210607-094856',
    'NVR-CH02_S20210607-173836_E20210607-173936',
    'NVR-CH02_S20210608-085112_E20210608-085718',
    'NVR-CH04_S20210608-083726_E20210608-083850',
    'NVR-CH04_S20210609-113009_E20210609-113658',
    'NVR-CH04_S20210609-182146_E20210609-182401',
    'NVR-CH06_S20210607-121005_E20210607-121221',
    'NVR-CH06_S20210608-111643_E20210608-111756',
    'NVR-CH06_S20210608-111915_E20210608-112037',
    'NVR-CH06_S20210609-082601_E20210609-083525',
    'NVR-CH06_S20210609-084956_E20210609-085058',
    'NVR-CH07_S20210609-084936_E20210609-085153'
]

# test_folder = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/test'

# new_test = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/test_after_split'
# new_val = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/val_after_split'

# def mkdir_if_missing(path):
#     if not os.path.exists(path):
#         print('Make dir {}'.format(path))
#         os.makedirs(path)


# mkdir_if_missing(new_val)
# mkdir_if_missing(new_test)

# for vid in val_vids:
#     try:
#         print('copy {} to val folder'.format(vid))
#         shutil.copytree(os.path.join(test_folder, vid), os.path.join(new_val, vid))
#         shutil.copy(os.path.join(test_folder, vid + '.mp4'), os.path.join(new_val, vid + '.mp4'))
#     except:
#         print('error vids: {}'.format(vid))


# for vid in test_vids:
#     try:
#         print('copy {} to test folder'.format(vid))
#         shutil.copytree(os.path.join(test_folder, vid), os.path.join(new_test, vid))
#         shutil.copy(os.path.join(test_folder, vid + '.mp4'), os.path.join(new_test, vid + '.mp4'))
#     except:
#         print('error vids: {}'.format(vid))

root = '/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/inference_tracking/detect_v3_reid_v8_cbdatav3/mot_preds'

for vid in val_vids:
    shutil.move(os.path.join(root, vid + '.txt'), os.path.join(root, 'val', vid + '.txt'))

for vid in test_vids:
    shutil.move(os.path.join(root, vid + '.txt'), os.path.join(root, 'test', vid + '.txt'))
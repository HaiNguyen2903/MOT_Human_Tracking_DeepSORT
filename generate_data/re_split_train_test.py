import os
from posixpath import dirname 
import shutil

frames_train = '/data.local/hangd/data_vtx/frames_data/train'
frames_test = '/data.local/hangd/data_vtx/frames_data/test'

labels_train = '/data.local/hangd/data_vtx/labels_yolo_format/train'
labels_test = '/data.local/hangd/data_vtx/labels_yolo_format/test'


old_train_vids = ["NVR-CH02_S20210607-173836_E20210607-173936.mp4",
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


old_test_vids = ["NVR-CH02_S20210607-094253_E20210607-094604.mp4",
             "NVR-CH02_S20210607-094604_E20210607-094856.mp4",
             "NVR-CH02_S20210607-170441_E20210607-170551.mp4",
             "NVR-CH02_S20210608-110218_E20210608-110328.mp4",
             "NVR-CH02_S20210608-120341_E20210608-120510.mp4",
             "NVR-CH02_S20210608-182322_E20210608-182524.mp4",
             "NVR-CH02_S20210609-172459_E20210609-172604.mp4",
             "NVR-CH02_S20210609-172604_E20210609-173139.mp4"
             ]

new_train_vids = [
    'NVR-CH02_S20210607-094253_E20210607-094604.mp4',
    'NVR-CH02_S20210607-170441_E20210607-170551.mp4',
    'NVR-CH02_S20210608-083623_E20210608-083906.mp4',
    'NVR-CH02_S20210608-084321_E20210608-084631.mp4',
    'NVR-CH02_S20210608-085752_E20210608-085950.mp4',
    'NVR-CH02_S20210608-110218_E20210608-110328.mp4',
    'NVR-CH02_S20210608-114057_E20210608-114337.mp4',
    'NVR-CH02_S20210608-114801_E20210608-115058.mp4',
    'NVR-CH02_S20210608-115057_E20210608-120139.mp4',
    'NVR-CH02_S20210608-120341_E20210608-120510.mp4',
    'NVR-CH02_S20210609-083605_E20210609-083710.mp4',
    'NVR-CH02_S20210609-083729_E20210609-083949.mp4',
    'NVR-CH02_S20210609-085200_E20210609-085751.mp4',
    'NVR-CH02_S20210609-172459_E20210609-172604.mp4',
    'NVR-CH02_S20210609-172604_E20210609-173139.mp4',
    'NVR-CH02_S20210609-173131_E20210609-173358.mp4',
    'NVR-CH02_S20210610-084425_E20210610-085713.mp4'
]

new_test_vids = [
    'NVR-CH02_S20210607-094604_E20210607-094856.mp4',
    'NVR-CH02_S20210607-173836_E20210607-173936.mp4',
    'NVR-CH02_S20210608-085112_E20210608-085718.mp4',
    'NVR-CH02_S20210608-114346_E20210608-114508.mp4',
    'NVR-CH02_S20210608-182322_E20210608-182524.mp4',
    'NVR-CH02_S20210609-115107_E20210609-115831.mp4'
]

old_train_names = [vid[:-4] for vid in old_train_vids]
old_test_names = [vid[:-4] for vid in old_test_vids]

new_train_names = [vid[:-4] for vid in new_train_vids]
new_test_names = [vid[:-4] for vid in new_test_vids]


# train
for vid in os.listdir(frames_train):
    if vid not in new_train_names:
        shutil.move(os.path.join(frames_train, vid), os.path.join(frames_test, vid))
        shutil.move(os.path.join(labels_train, vid), os.path.join(labels_test, vid))

# test
for vid in os.listdir(frames_test):
    if vid not in new_test_names:
        shutil.move(os.path.join(frames_test, vid), os.path.join(frames_train, vid))
        shutil.move(os.path.join(labels_test, vid), os.path.join(labels_train, vid))



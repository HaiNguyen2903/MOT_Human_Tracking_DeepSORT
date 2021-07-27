import os 

src_frames_dir = '/data.local/hangd/data_vtx/frames_data/test/NVR-CH07_S20210609-084936_E20210609-085153'
src_labels_dir = '/data.local/hangd/data_vtx/detection_dataset/labels/test_gt/NVR-CH07_S20210609-084936_E20210609-085153'

toy_imgs = '/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/test_dataset/images'

toy_labels = '/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/test_dataset/labels'


for img in os.listdir(src_frames_dir):
    os.symlink(os.path.join(src_frames_dir, img), os.path.join(toy_imgs, img[:-3] + 'jpg'))

for label in os.listdir(src_labels_dir):
    os.symlink(os.path.join(src_labels_dir, label), os.path.join(toy_labels, label))

assert len(os.listdir(toy_imgs)) == len(os.listdir(toy_labels))

print('Length toy dataset:', len(os.listdir(toy_imgs)))


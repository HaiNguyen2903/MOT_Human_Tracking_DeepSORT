import os

# root = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/test'
# for dir in os.listdir(root):
#     if os.path.isdir(os.path.join(root,dir)):
#         print(dir)
#         for subdir in os.listdir(os.path.join(root, dir)):
#             if os.path.isdir(os.path.join(root, dir, subdir)):
#                 print(subdir)
#         print()

# labels = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset_v3/detection_dataset/labels/test'
# images = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset_v3/detection_dataset/images/test'

# print(len(os.listdir(labels)))
# print(len(os.listdir(images)))
print(len(os.listdir('/data.local/hangd/data_vtx/mot_annotations/annotationCH02_reviewed')))

# print()

# label_ls = []
# for img in sorted(os.listdir(labels)):
#     label_ls.append(int(img[6:12]))
#     print(img[6:12])

# print('Label len: {}'.format(len(os.listdir(labels))))

# for i in range(1, len(label_ls) - 1):
#     print(label_ls[i] - label_ls[i-1])

# print(len(os.listdir('/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset_v3/detection_dataset/labels_by_vids/train/NVR-CH01_S20210607-094253_E20210607-094856')))

# print(len(os.listdir('/data.local/hangd/human_tracking/fairmot_human_tracking/dataset/UET_MOT_dataversion3+4/images/train/NVR-CH01_S20210607-094253_E20210607-094856/img1')))
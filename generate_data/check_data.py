import os

root = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET/test'
for dir in os.listdir(root):
    if os.path.isdir(os.path.join(root,dir)):
        print(dir)
        for subdir in os.listdir(os.path.join(root, dir)):
            if os.path.isdir(os.path.join(root, dir, subdir)):
                print(subdir)
        print()
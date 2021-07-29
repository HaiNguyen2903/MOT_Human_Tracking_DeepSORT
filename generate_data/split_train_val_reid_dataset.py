import shutil
import os
from shutil import copyfile

train_dir = '/data.local/hangd/data_vtx/reid_dataset/uet_reid/train'
test_dir = '/data.local/hangd/data_vtx/reid_dataset/uet_reid/test'

def count_frames(root_dir):
    count = 0
    for dir in os.listdir(root_dir):
        count += len(os.listdir(os.path.join(root_dir, dir)))
    print(count)
    return count

len_train = len(os.listdir(train_dir))

ratio = 0.1

len_val = int(len_train * ratio)

def split_train_test(train_dir, test_dir, count=0):
    for dir in os.listdir(train_dir):
        if count <= len_val:
            src = os.path.join(train_dir, dir)
            desc = os.path.join(test_dir, dir)

            print('Copying {}'.format(src))

            shutil.copytree(src, desc)
            # shutil.rmtree(src)

            count += 1

    print('Done')

split_train_test(train_dir, test_dir)

train_size = count_frames(train_dir)
test_size = count_frames(test_dir)

print('Ratio', test_size/train_size)
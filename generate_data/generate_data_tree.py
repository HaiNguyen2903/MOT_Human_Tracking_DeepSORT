from genericpath import exists
import os
import glob


data_root = '/data.local/hangd/full_data_vtx_backup/data_vtx/data'

gt_root = '/data.local/hangd/data_vtx/mot_annotations/annotation_CH06_reviewed/'

tree_root = '/data.local/hangd/data_vtx/DATA_ROOT/data_version4/TRAIN_DATASET'


train_vids = [
    # 'NVR-CH04_S20210608-173010_E20210608-173657.mp4',
    # 'NVR-CH04_S20210608-173657_E20210608-180358.mp4',
    # 'NVR-CH04_S20210608-180358_E20210608-182039.mp4',
    # 'NVR-CH04_S20210609-175447_E20210609-182146.mp4'
    'NVR-CH06_S20210608-081721_E20210608-082317.mp4',
    'NVR-CH06_S20210608-112708_E20210608-112828.mp4',
    'NVR-CH06_S20210608-112822_E20210608-112924.mp4',
    'NVR-CH06_S20210608-112822_E20210608-112924.mp4',
    'NVR-CH06_S20210608-112929_E20210608-113357.mp4',


]

test_vids = [
    # 'NVR-CH04_S20210608-083726_E20210608-083850.mp4',
    # 'NVR-CH04_S20210609-113009_E20210609-113658.mp4',
    # 'NVR-CH04_S20210609-173203_E20210609-175447.mp4',
    # 'NVR-CH04_S20210609-182146_E20210609-182401.mp4'
    'NVR-CH06_S20210607-121005_E20210607-121221.mp4',
    'NVR-CH06_S20210607-131815_E20210607-131952.mp4',
    'NVR-CH06_S20210608-111643_E20210608-111756.mp4',
    'NVR-CH06_S20210608-111753_E20210608-111910.mp4',
    'NVR-CH06_S20210608-111915_E20210608-112037.mp4',
    'NVR-CH06_S20210608-112537_E20210608-112642.mp4',
    'NVR-CH06_S20210609-082601_E20210609-083525.mp4',
    'NVR-CH06_S20210609-084956_E20210609-085058.mp4',
    'NVR-CH06_S20210609-111136_E20210609-111336.mp4',
    'NVR-CH06_S20210609-111615_E20210609-111734.mp4'
]

train_names = [vid[:-4] for vid in train_vids]
test_names = [vid[:-4] for vid in test_vids]


def mkdir_if_missing(full_path):
    if not os.path.exists(full_path):
        print('Create dir {}'.format(full_path))
        os.makedirs(full_path)


def search_paths(data_root, gt_root, name):
    vid_paths = glob.glob(os.path.join(data_root, '**/*', name + '.mp4'))
    gt_paths = glob.glob(os.path.join(gt_root, name))
    
    if len(vid_paths) > 0 and len(gt_paths) > 0:
        return vid_paths[0], gt_paths[0]

    else:
        if len(vid_paths) == 0:
            print('Missing video {}'.format(name))
        else:
            print('Missing gt for video {}'.format(name))
        return None, None



def gen_tree_data(tree_root, data_root, gt_root, train_names, test_names):
    train_missing = []
    test_missing = []

    mkdir_if_missing(tree_root)

    for name in train_names:
        full_path = os.path.join(tree_root, 'train')
        mkdir_if_missing(full_path)

        vid_src, gt_src = search_paths(data_root, gt_root, name)

        if vid_src == None or gt_src == None:
            train_missing.append(name)
        else:
            vid_dest = os.path.join(full_path, name + '.mp4')
            gt_dest = os.path.join(full_path, name)

            if not os.path.exists(vid_dest):
                os.symlink(vid_src, vid_dest) 
                print('Symlink video {}'.format(name))
            if not os.path.exists(gt_dest): 
                os.symlink(gt_src, gt_dest)
                print('Symlink gt for video {}'.format(name))

    for name in test_names:
        full_path = os.path.join(tree_root, 'test')
        mkdir_if_missing(full_path)

        vid_src, gt_src = search_paths(data_root, gt_root, name)

        if vid_src == None or gt_src == None:
            test_missing.append(name)
        else:
            vid_dest = os.path.join(full_path, name + '.mp4')
            gt_dest = os.path.join(full_path, name)

            if not os.path.exists(vid_dest):
                os.symlink(vid_src, vid_dest) 
                print('Symlink video {}'.format(name))
            if not os.path.exists(gt_dest): 
                os.symlink(gt_src, gt_dest)
                print('Symlink gt for video {}'.format(name))

    return train_missing, test_missing


train_missing, test_missing = gen_tree_data(tree_root, data_root, gt_root, train_names, test_names)

print()

print('Train missing')
print(train_missing)
print()
print('Test missing')
print(test_missing)
# print(train_paths[0])
# print(train_paths)
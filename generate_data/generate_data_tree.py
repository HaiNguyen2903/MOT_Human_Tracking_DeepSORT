from genericpath import exists
import os
import glob


# data_root = '/data.local/hangd/full_data_vtx_backup/data_vtx/data'

# path on 1080Ti
data_root = '/data/data_vtx/data'

# gt_root = '/data.local/hangd/data_vtx/mot_annotations/annotationCH02_reviewed/'

# path on 1080Ti
gt_root = '/data/DATA_ROOT/data_version4/gt/annotation_CH05_reviewed'


# tree_root = '/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/TRAIN_DATASET'

tree_root = '/data/DATA_ROOT/data_version4/TRAIN_DATASET'


train_vids = [
    # 'NVR-CH02_S20210607-094253_E20210607-094604.mp4',
    # 'NVR-CH02_S20210607-170441_E20210607-170551.mp4',
    # 'NVR-CH02_S20210608-083623_E20210608-083906.mp4',
    # 'NVR-CH02_S20210608-084321_E20210608-084631.mp4',
    # 'NVR-CH02_S20210608-085752_E20210608-085950.mp4',
    # 'NVR-CH02_S20210608-110218_E20210608-110328.mp4',
    # 'NVR-CH02_S20210608-114057_E20210608-114337.mp4',
    # 'NVR-CH02_S20210608-114801_E20210608-115058.mp4',
    # 'NVR-CH02_S20210608-115057_E20210608-120139.mp4',
    # 'NVR-CH02_S20210608-120341_E20210608-120510.mp4',
    # 'NVR-CH02_S20210609-083605_E20210609-083710.mp4',
    # 'NVR-CH02_S20210609-083729_E20210609-083949.mp4',
    # 'NVR-CH02_S20210609-085200_E20210609-085751.mp4',
    # 'NVR-CH02_S20210609-172459_E20210609-172604.mp4',
    # 'NVR-CH02_S20210609-172604_E20210609-173139.mp4',
    # 'NVR-CH02_S20210609-173131_E20210609-173358.mp4',
    # 'NVR-CH02_S20210610-084425_E20210610-085713.mp4'
]

test_vids = [
    # 'NVR-CH02_S20210607-094604_E20210607-094856.mp4',
    # 'NVR-CH02_S20210607-173836_E20210607-173936.mp4',
    # 'NVR-CH02_S20210608-085112_E20210608-085718.mp4',
    # 'NVR-CH02_S20210608-114346_E20210608-114508.mp4',
    # 'NVR-CH02_S20210608-182322_E20210608-182524.mp4',
    # 'NVR-CH02_S20210609-115107_E20210609-115831.mp4',

    # 'NVR-CH08_S20210608-084003_E20210608-084414.mp4',
    # 'NVR-CH08_S20210608-112806_E20210608-113059.mp4',
    # 'NVR-CH08_S20210608-113115_E20210608-113339.mp4',
    # 'NVR-CH08_S20210609-111643_E20210609-111832.mp4',
    # 'NVR-CH08_S20210609-174751_E20210609-175009.mp4',
    # 'NVR-CH08_S20210610-074843_E20210610-075118.mp4',
    # 'NVR-CH08_S20210610-082003_E20210610-082129.mp4'

    'NVR-CH05_S20210608-081616_E20210608-081841.mp4',
    'NVR-CH05_S20210608-082702_E20210608-083122.mp4',
    'NVR-CH05_S20210608-170913_E20210608-171111.mp4',
    'NVR-CH05_S20210609-120054_E20210609-120216.mp4',
    'NVR-CH05_S20210610-080844_E20210610-081006.mp4',
    'NVR-CH05_S20210610-082053_E20210610-082315.mp4'
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
import os
from os import listdir as osl
from os import path as osp
from IPython import embed

'''
This file seperates the original gallery into sub - gallery for each query using symlink
A query at frame x is matched with a gallery in range (x - bound, x + bound) (bound = 100 by default)
'''

combine_gallery_dir  = '/data.local/hangd/data_vtx/reid_dataset/uet_reid/gallery'
query_dir = '/data.local/hangd/data_vtx/reid_dataset/uet_reid/query'
save_root = '/data.local/hangd/data_vtx/reid_dataset/uet_reid/sub_galleries'


def mkdir_if_missing(path):
    if not os.path.exists(path):
        print('Make dir {}'.format(path))
        os.mkdir(path)


def cal_total_gallery_images(combine_gallery_dir):
    count = 0
    for dir in osl(combine_gallery_dir):
        count += len(osl(osp.join(combine_gallery_dir, dir)))

    print('Total gallery images: {}'.format(count))

    return count


total = cal_total_gallery_images(combine_gallery_dir)


def generate_sub_galleries(combine_gallery_dir, query_dir, save_root, range):
    mkdir_if_missing(save_root)

    q_frames = {}

    # get all query frame ids and create sub gallery folders
    for query in sorted(osl(query_dir)):
        for q in sorted(osl(osp.join(query_dir, query))):
            # get frame id of the image (123.jpg)
            frame_id = int(q[:-4])
            mkdir_if_missing(osp.join(save_root, 'gallery_{}'.format(str(query))))
            q_frames[query] = frame_id


    # for each id folder in gallery
    for sub_dir in sorted(osl(combine_gallery_dir)):
        # for each id in current folder 
        for id in sorted(osl(osp.join(combine_gallery_dir, sub_dir))):
            print(id)
            g_frame = int(id[:-4])

            for q_id in q_frames:
                lower_bound = max(0, q_frames[q_id] - range)
                # id start from 0
                upper_bound = min(total - 1, q_frames[q_id] + range)

                # embed()
                # if g_id in range(lower_bound, upper_bound + 1):
                if lower_bound <= g_frame <= upper_bound:
                    src_img = osp.join(combine_gallery_dir, sub_dir, id)

                    dest_img = osp.join(save_root, 'gallery_{}'.format(str(q_id)), sub_dir, id)

                    mkdir_if_missing(osp.join(save_root, 'gallery_{}'.format(str(q_id)), sub_dir))

                    print('Symlink gallery for query id {}'.format(q_id))
                    os.symlink(src_img, dest_img)

def check_data(query_dir, sub_gallery_root):
    '''
    check already generated sub galleries data
    '''
    
    return
        
generate_sub_galleries(combine_gallery_dir, query_dir, save_root, range=100)
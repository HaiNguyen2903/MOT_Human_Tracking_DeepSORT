import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gt-path', default = '../track_dataset/gt.txt', type = str, help = 'path to MOT grouth truth path')
parser.add_argument('--save-path', default = '../track_dataset/gt_yolo_format', type = str, help = 'path to output save dir')

args = parser.parse_args()


gt_file = args.gt_path
assert os.path.isfile(gt_file)

save_dir = args.save_path

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def reformat_gt(gt_file, save_dir):
    file = open(gt_file, 'r')
    lines = file.readlines()

    saved_frames = []
    prev_frame = 0

    for line in lines:
        line = line.strip()
        ann = line.split(',')

        current_frame = ann[0]

        if current_frame != prev_frame:
            print('Writing frame {} ground truths'.format(current_frame))
            if current_frame not in saved_frames:
                saved_frames.append(current_frame)

            out = open(os.path.join(save_dir, 'frame_{:06d}.txt'.format(int(current_frame)-1)), 'w')
            out.write('person {} {} {} {}\n'.format(ann[2], ann[3], str(float(ann[2]) + float(ann[4])), str(float(ann[3]) + float(ann[5]))))
            out.close()
        else:
            out = open(os.path.join(save_dir, 'frame_{:06d}.txt'.format(int(current_frame)-1)), 'a')

            out.write('person {} {} {} {}\n'.format(ann[2], ann[3], str(float(ann[2]) + float(ann[4])), str(float(ann[3]) + float(ann[5]))))
            out.close()
        prev_frame = current_frame

    # handling frame contains no object
    print('Handling lost frames')
    ids = []
    for i in range(1, 200):
        ids.append(i)

    for id in ids:
        if str(id) not in saved_frames:
            out = open(os.path.join(save_dir, 'frame_{:06d}.txt'.format(id-1)), 'w')
    print('Finishing extracting ground truths.')

reformat_gt(gt_file=gt_file, save_dir=save_dir)
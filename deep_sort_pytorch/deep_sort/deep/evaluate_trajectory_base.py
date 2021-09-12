from re import A
import torch
from IPython import embed
import argparse
import os
import cv2
import numpy as np
from os import path as osp
import json

def get_frame_index(path):
    return int(os.path.basename(path)[:-4])


def get_dirname(path):
    dir_path = osp.dirname(path)
    return int(osp.base(dir_path))


def get_trajectory_bounds(gl, gallery_paths, limit=100):
    print('Getting start and end frame for all trajectories ...')
    # max frame in the gallery
    max_frame = 0

    # trajectory_bound[trajectory_id] = {upper_bound: upper_bound, lower_bound: lower_bound}
    trajectory_bounds = {}

    # get max frame in gallery
    for path in gallery_paths:
        gframe = get_frame_index(path)
        if gframe > max_frame:
            max_frame = gframe

    # embed()

    for i in range(gl.size(0)):
        pid = gl[i].item()

        # print('Getting start and end frame for trajectory of id {}'.format(pid))

        gframe = get_frame_index(gallery_paths[i])

        if pid not in trajectory_bounds:
            lower_bound = max(0, gframe - limit)
            upper_bound = min(max_frame, gframe + limit)

            trajectory_bounds[pid] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}

        else:
            trajectory_bounds[pid]['lower_bound'] = max(0, min(trajectory_bounds[pid]['lower_bound'], gframe - limit))
            trajectory_bounds[pid]['upper_bound'] = min(max_frame, max(trajectory_bounds[pid]['upper_bound'], gframe + limit))

    # embed(header='debug func')
    return trajectory_bounds


def get_pid_bound(test_json):
    pid_bound = {}

    with open(test_json) as json_file:
        test_info = json.load(json_file)
        for vid in test_info:
            start_pid, end_pid = test_info[vid][0], test_info[vid][1]
            for id in range(start_pid, end_pid + 1):
                pid_bound[id] = [start_pid, end_pid]

    return pid_bound


def collect_features(qf, ql, gf, gl, query_paths, gallery_paths, limit, min_frame, test_json):
    print('Calculating features for each query ...') 
    print()

    # list of features for each query
    features = []
    # number of ignore cases (does not satisfy evaluation requirement)
    ignores = 0

    trajectory_bounds = get_trajectory_bounds(gl, gallery_paths, limit=limit)
    pid_bounds = get_pid_bound(test_json)

    # for each query 
    for i in range(qf.size(0)):
        pid = ql[i].item()

        # vector feature, label and path to image for each query
        q_feature = torch.tensor([]).float()
        q_label = torch.tensor([]).long()
        q_path = []

        # vector features, labels and paths to images in gallery
        g_features = torch.tensor([]).float()
        g_labels = torch.tensor([]).long()
        g_paths = []

        # embed(header='debug')

        # update info of query
        q_feat = torch.unsqueeze(qf[i], dim=0)
        q_feature = torch.cat((q_feature, q_feat), dim = 0)
        q_label = torch.cat((q_label, torch.LongTensor([ql[i]])))
        q_path.append(query_paths[i])

        # get frame index of current query
        q_frame = get_frame_index(q_path[0])

        # calculate lower bound and upper bound of the related gallery
        lower_bound = trajectory_bounds[pid]['lower_bound']
        # upper_bound = min(gf.size(0), q_frame + limit)
        upper_bound = trajectory_bounds[pid]['upper_bound']

        start_pid = pid_bounds[pid][0]
        end_pid = pid_bounds[pid][1]

        ## check all instances in gallery that in range [lower_bound, upper_bound] and update
        # for i in range(gf.size(0)):
        #     g_path = gallery_paths[i]
        #     g_frame = get_frame_index(g_path)
        #     if lower_bound <= g_frame <= upper_bound:
        #         # embed(header='debug gallery feature')
        #         g_feat = torch.unsqueeze(gf[i], dim=0)
        #         g_features = torch.cat((g_features, g_feat), dim = 0)
        #         g_labels = torch.cat((g_labels, torch.LongTensor([gl[i]])))
        #         g_paths.append(gallery_paths[i])

        # using matrix for faster calculation with torch

        # embed(header='debug collect features')
        g_paths = np.array([g_path for g_path in gallery_paths])
        g_frames = np.array([get_frame_index(g_path) for g_path in g_paths])
        g_pids = np.array(gl)

        valid_idx = np.where((lower_bound <= g_frames) & (g_frames <= upper_bound) & (start_pid <= g_pids) & (g_pids <= end_pid))[0]
        
        g_features = gf[valid_idx]
        g_labels = gl[valid_idx]
        # filter path with valid idx
        g_paths = g_paths[valid_idx]

        # print('query id: {}  gallery id: {}'.format(pid, g_labels))

        # check if a gallery is satisfy the requirement or not
        if len(g_paths) < min_frame:
            ignores += 1
            continue
        
        # update features list 
        feat = {
            "qf": q_feature,
            "ql": q_label,
            "gf": g_features,
            "gl": g_labels,
            "query_paths": q_path,
            "gallery_paths": g_paths
        }

        features.append(feat)

    # embed(header='debug collect features')

    return features, ignores


def calculate_rank_1(features):

    mean_top1_acc = 0

    for feat in features:
        # embed(header='debug rank 1')
        qf = feat["qf"]
        ql = feat["ql"]
        gf = feat["gf"]
        gl = feat["gl"]

        score = qf.mm(gf.t())

        # return vector of index in scores metric that have highest score for each query: len = query frame
        res = score.topk(1, dim=1)[1][:, 0]

        # total equal values between ql and ql vectors
        top1correct = gl[res].eq(ql).sum().item()

        top1acc = top1correct / ql.size(0)

        mean_top1_acc += top1acc
    
    mean_top1_acc = (mean_top1_acc / len(features)) 

    # print("Accuracy top 1: {:.3f}".format(top1correct / ql.size(0)))
    print("Accuracy top 1: {:.3f}".format(mean_top1_acc*100))
    return mean_top1_acc


def calculate_precision_k(features, k):
    avg_acc = 0
    # return vector of index in scores metric that have highest score for each query: len = query frame

    for feat in features:
        # embed(header='debug rank 1')
        qf = feat["qf"]
        ql = feat["ql"]
        gf = feat["gf"]
        gl = feat["gl"]

        score = qf.mm(gf.t())

        # return vector of index in scores metric that have highest score for each query: len = query frame
        res = score.topk(k, dim=1)[1]

        # top_pred for each query, return matrix of number query x k
        pred = gl[res]

        # count number of label value in top k pred equal to true label
        correct = pred[0].eq(ql[0]).sum().item()

        acc = correct / k

        avg_acc += acc

    avg_acc = (avg_acc / len(features)) 
    print('P@{}: {:.3f}'.format(k, avg_acc*100))

    return avg_acc


'''
Calculate precison@k and AP@n for each query instead of average on all queries
'''

def calculate_mAP_n(features, n):
    mAP = 0

    for feat in features:
        qf = feat["qf"]
        ql = feat["ql"]
        gf = feat["gf"]
        gl = feat["gl"]

        score = qf.mm(gf.t())

        res = score.topk(n, dim=1)[1]
        # top n pred for each query, return matrix of (number query x n)
        pred = gl[res]

        total_correct = pred[0].eq(ql[0]).sum().item()

        AP = 0

        for k in range(1, n+1):
            # top k pred for each query
            pred_k = pred[:, :k]
            
            equal = pred_k[0].eq(ql[0])

            # if the image at rank k is relevant, then add it to AP calculation, else skip
            if equal[k-1]:
                # number of correct in top k
                correct_k = 0
                correct_k += pred_k[0].eq(ql[0]).sum().item()
                
                # calculate p@k
                precision_k = correct_k / k

                # adding to overall AP of this query
                AP += precision_k / total_correct
    
        # embed()
        # print()
        # print()
        # add AP to calculate mAP on all queries
        mAP += AP 

    mAP = (mAP / len(features)) 
    # print(pred_k.shape)
    print('mAP@{}: {:.3f}'.format(n, mAP*100))
    return mAP


'''
Visualize rank k result for each query
'''

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def mkdir_if_missing(path):
    if not os.path.exists(path):
        print('Make dir {}'.format(path))
        os.mkdir(path)


def visualize_rank_k(features, output_dir, topk, width=128, height=256):
    mkdir_if_missing(output_dir)

    for feat in features:
        qf = feat["qf"]
        ql = feat["ql"]
        gf = feat["gf"]
        gl = feat["gl"]

        query_paths = feat['query_paths']
        gallery_paths = feat['gallery_paths']

        # embed(header='deubg visualize')
        score = qf.mm(gf.t())

        # return vector of index in scores metric that have highest score for each query: len = query frame
        indices = score.topk(topk, dim=1)[1]
            # embed(header='debug visualize')

        # top k pred for each query, return matrix of (number query x n)
        pred = gl[indices]

    # for i in range(ql.size(0)): 
        qimg = cv2.imread(query_paths[0])
        qimg = cv2.resize(qimg, (width, height))
        qimg = cv2.copyMakeBorder(
            qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # resize twice to ensure that the border width is consistent across images
        qimg = cv2.resize(qimg, (width, height))

        q_frame = os.path.basename(query_paths[0])[:-4]
        print('query frame: {}'.format(q_frame))

        qimg = cv2.putText(img = qimg,
                            text = q_frame,
                            org = (10,25),
                            fontFace = cv2.FONT_HERSHEY_DUPLEX,
                            fontScale = 0.8,
                            color = (255, 255, 0),
                            thickness = 2)

        num_cols = topk + 1
        grid_img = 255 * np.ones(
            (
                height,
                num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
            ),
            dtype=np.uint8
        )
        grid_img[:, :width, :] = qimg


        # make subdir for each query
        pid = ql[0].item()
        mkdir_if_missing(os.path.join(output_dir, str(pid)))

        # list of matched predicts (boolean tensor)
        matches = pred[0].eq(ql[0])

        rank_idx = 1

        for j in range(len(matches)):
            border_color = GREEN if matches[j] else RED

            g_path = gallery_paths[indices[0][j].item()]
            g_frame = os.path.basename(g_path)[:-4]
            print('gallery frame: {}'.format(g_frame))

            gimg = cv2.imread(g_path)
            gimg = cv2.resize(gimg, (width, height))
            gimg = cv2.copyMakeBorder(
                        gimg,
                        BW,
                        BW,
                        BW,
                        BW,
                        cv2.BORDER_CONSTANT,
                        value=border_color
                    )

            gimg = cv2.resize(gimg, (width, height))

            gimg = cv2.putText(img = gimg,
                               text = g_frame,
                               org = (10,25),
                               fontFace = cv2.FONT_HERSHEY_DUPLEX,
                               fontScale = 0.8,
                               color = (255, 255, 0),
                               thickness = 2)

            start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
            end = (
                rank_idx+1
            ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:, start:end, :] = gimg

            rank_idx += 1
            if rank_idx > topk:
                break
        
        print()
    
        imname = os.path.basename(query_paths[0])[:-4]
        cv2.imwrite(os.path.join(output_dir, str(pid), imname + '.jpg'), grid_img)
        # print('Writing {}.jpg'.format(imname))
    
    print()
    print('Successfully saving inference images.')


def evaluate(features, p_k, mAP_n, range, show = False, visualize_topk=5, infer_dir = None, test_json=None):
    '''
    gf: query frames: shape (frames x features_len) (208 x 512)
    ql: query labels: vector len = number of query images
    gf: gallery frames: shape (frames x features_len) (21549 x 512)
    gl: gallery labels: vector len = number of gallery images
    '''

    qf = features["qf"]
    ql = features["ql"]
    gf = features["gf"]
    gl = features["gl"]

    query_paths = features['query_paths']
    gallery_paths = features['gallery_paths']

    # if visualize top k
    if show:
        min_frame = max(p_k, mAP_n, visualize_topk)
    else:
        min_frame = max(p_k, mAP_n)

    # calculate feature base on 
    feat_list, ignores = collect_features(qf, ql, gf, gl, query_paths, gallery_paths, limit=range, min_frame=min_frame, test_json=test_json)

    acc_top1 = calculate_rank_1(feat_list)
    precision_k = calculate_precision_k(feat_list, p_k)
    mAP = calculate_mAP_n(feat_list, mAP_n)

    print()
    print('Number of ignore queries: {}'.format(ignores))
    if show:
        visualize_rank_k(feat_list, topk = visualize_topk, output_dir = infer_dir)

    return acc_top1, precision_k, mAP



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--predict_path", default='predicts/features_new.pth', type=str)
    parser.add_argument("--p_k", default=5, type=int)
    parser.add_argument("--mAP_n", default=5, type=int)
    parser.add_argument("--range", default=100, type=int, help='evaluate in range [x-range, x+ range] for frame x')
    # parser.add_argument("--frames_require", default=10, type=int, help='min number of frame require in gallery for evaluating')
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--visualize_top_k", default=10, type=int)
    parser.add_argument("--inference_dir", default = "inference_test", type=str)

    parser.add_argument("--train_json", default='/data.local/hangd/human_tracking/ALL_SCRIPTS/generate_reid_data/uet_reid/train_video_infos.json', type=str)
    parser.add_argument("--test_json", default='/data.local/hangd/human_tracking/ALL_SCRIPTS/generate_reid_data/uet_reid/test_video_infos.json', type=str)

    args = parser.parse_args()

    # train_info = json.loads(args.train_json)
    # with open(args.train_json) as json_file:
    #     data = json.load(json_file)

    #     embed()

    features = torch.load(args.predict_path)

    evaluate(features = features,
            p_k = args.p_k,
            mAP_n = args.mAP_n,
            range = args.range,
            show = args.show,
            visualize_topk = args.visualize_top_k,
            infer_dir = args.inference_dir,
            test_json=args.test_json)
    
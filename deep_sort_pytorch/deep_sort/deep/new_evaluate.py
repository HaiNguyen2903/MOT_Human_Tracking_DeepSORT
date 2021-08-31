from re import A
import torch
from IPython import embed
import argparse
import os
import cv2
import numpy as np

# parser = argparse.ArgumentParser(description="Train on market1501")
# parser.add_argument("--predict_path", default='predicts/features_new.pth', type=str)
# parser.add_argument("--p_k", default=5, type=int)
# parser.add_argument("--map_n", default=5, type=int)
# parser.add_argument("--show", action='store_true')
# parser.add_argument("--visualize_rank_k", default=10, type=int)
# parser.add_argument("--inference_dir", default = "inference_test", type=str)

# args = parser.parse_args()

# features = torch.load(args.predict_path)

# '''
# gf: query frames: shape (frames x features_len) (208 x 512)
# ql: query labels: vector len = number of query images
# gf: gallery frames: shape (frames x features_len) (21549 x 512)
# gl: gallery labels: vector len = number of gallery images
# '''

# qf = features["qf"]
# ql = features["ql"]
# gf = features["gf"]
# gl = features["gl"]

# query_paths = features['query_paths']
# gallery_paths = features['gallery_paths']


def cal_query_scores(qf, gf, query_paths, gallery_paths, limit, min_frame):
    print('Calculating features for each query ...') 
    print()
    # dict of info to eval for each query where key is query_id and value are features for that query
    features = []
    ignores = 0

    # for each query 
    for i in range(qf.size(0)):
        q_feature = torch.tensor([]).float()
        q_label = torch.tensor([]).long()
        q_path = []

        g_features = torch.tensor([]).float()
        g_labels = torch.tensor([]).long()
        g_paths = []

        # embed(header='debug')

        # query_features = torch.cat((query_features, features), dim=0)
        # query_labels = torch.cat((query_labels, labels))
        # query_paths.extend(paths)

        q_feat = torch.unsqueeze(qf[i], dim=0)
        q_feature = torch.cat((q_feature, q_feat), dim = 0)
        q_label = torch.cat((q_label, torch.LongTensor([ql[i]])))
        q_path.append(query_paths[i])

        q_frame = int(os.path.basename(q_path[0])[:-4])
        
        lower_bound = max(0, q_frame - limit)
        upper_bound = min(gf.size(0), q_frame + limit)

        for i in range(gf.size(0)):
            g_path = gallery_paths[i]
            g_frame = int(os.path.basename(g_path)[:-4])
            if lower_bound <= g_frame <= upper_bound:
                # embed(header='debug gallery feature')
                g_feat = torch.unsqueeze(gf[i], dim=0)
                g_features = torch.cat((g_features, g_feat), dim = 0)
                g_labels = torch.cat((g_labels, torch.LongTensor([gl[i]])))
                g_paths.append(gallery_paths[i])

        # check if a gallery is satisfy the requirement or not
        if len(g_paths) < min_frame:
            ignores += 1
            continue

        feat = {
            "qf": q_feature,
            "ql": q_label,
            "gf": g_features,
            "gl": g_labels,
            "query_paths": q_path,
            "gallery_paths": g_paths
        }

        features.append(feat)
            
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
    
    mean_top1_acc /= len(features)

    # print("Accuracy top 1: {:.3f}".format(top1correct / ql.size(0)))
    print("Accuracy top 1: {:.3f}".format(mean_top1_acc))
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

        # embed(header = 'debug pk')

        # top_pred for each query, return matrix of number query x k
        pred = gl[res]

        # count number of label value in top k pred equal to true label
        correct = pred[0].eq(ql[0]).sum().item()

        acc = correct / k

        avg_acc += acc

    avg_acc = avg_acc / len(features)
    print('P@{}: {:.3f}'.format(k, avg_acc))

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

    mAP /= len(features)
    # print(pred_k.shape)
    print('mAP@{}: {:.3f}'.format(n, mAP))
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
            gimg = cv2.imread(gallery_paths[indices[0][j].item()])
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
            start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
            end = (
                rank_idx+1
            ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:, start:end, :] = gimg

            rank_idx += 1
            if rank_idx > topk:
                break

    
        imname = os.path.basename(query_paths[0])[:-4]
        cv2.imwrite(os.path.join(output_dir, str(pid), imname + '.jpg'), grid_img)
        # print('Writing {}.jpg'.format(imname))
    
    print()
    print('Successfully saving inference images.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--predict_path", default='predicts/features_new.pth', type=str)
    parser.add_argument("--p_k", default=5, type=int)
    parser.add_argument("--map_n", default=5, type=int)
    parser.add_argument("--range", default=100, type=int, help='evaluate in range [x-range, x+ range] for frame x')
    # parser.add_argument("--frames_require", default=10, type=int, help='min number of frame require in gallery for evaluating')
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--visualize_rank_k", default=10, type=int)
    parser.add_argument("--inference_dir", default = "inference_test", type=str)

    args = parser.parse_args()

    features = torch.load(args.predict_path)

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

    # calculate feature base on 
    feat_list, ignores = cal_query_scores(qf, gf, query_paths, gallery_paths, limit=args.range, min_frame = min(args.p_k, args.map_n))

    calculate_rank_1(feat_list)
    calculate_precision_k(feat_list, args.p_k)
    calculate_mAP_n(feat_list, args.map_n)

    print()
    print('Number of ignore queries: {}'.format(ignores))
    if args.show:
        visualize_rank_k(feat_list, topk = args.visualize_rank_k, output_dir = args.inference_dir)



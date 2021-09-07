import torch
from IPython import embed
import argparse
import os
import cv2
import numpy as np

def calculate_rank_1(scores, features):
    qf = features["qf"]
    ql = features["ql"]
    gf = features["gf"]
    gl = features["gl"]

    # return vector of index in scores metric that have highest score for each query: len = query frame
    res = scores.topk(1, dim=1)[1][:, 0]

    # total equal values between ql and ql vectors
    top1correct = gl[res].eq(ql).sum().item()
    
    print("Accuracy top 1: {:.5f}".format(top1correct / ql.size(0)))
    return (top1correct / ql.size(0))


def calculate_precision_k(scores, features, k=5):
    qf = features["qf"]
    ql = features["ql"]
    gf = features["gf"]
    gl = features["gl"]

    # return vector of index in scores metric that have highest score for each query: len = query frame
    res = scores.topk(k, dim=1)[1]

    # top_pred for each query, return matrix of number query x k
    pred = gl[res]

    # average p@k for all queries
    avg_acc = 0

    # for each query
    for i in range(ql.size(0)): 
        # count number of label value in top k pred equal to true label
        correct = pred[i].eq(ql[i]).sum().item()

        # acc for each query
        acc = correct/k
        
        avg_acc += acc

    # calculate average acc
    avg_acc = (avg_acc / ql.size(0)) 

    print('P@{}: {:.5f}'.format(k, avg_acc))
    return avg_acc


'''
Calculate precison@k and AP@n for each query instead of average on all queries
'''

def calculate_mAP_n(scores, features, n=5):
    qf = features["qf"]
    ql = features["ql"]
    gf = features["gf"]
    gl = features["gl"]

    mAP = 0

    res = scores.topk(n, dim=1)[1]
    # top n pred for each query, return matrix of (number query x n)
    pred = gl[res]

    # for each query
    for i in range(ql.size(0)):
        # total correct in top n
        # total_correct = 0
        total_correct = pred[i].eq(ql[i]).sum().item()

        AP = 0

        # for k from 1 to n
        for k in range(1, n+1):
            # top k pred for each query
            pred_k = pred[:, :k]

            equal = pred_k[i].eq(ql[i])

            # if the image at rank k is relevant, then add it to AP calculation, else skip
            if equal[k-1]:
                # number of correct in top k
                correct_k = 0
                correct_k += pred_k[i].eq(ql[i]).sum().item()
                
                # calculate p@k
                precision_k = correct_k / k

                # adding to overall AP of this query
                AP += precision_k / total_correct

                # debug
                # print('k: {}  total correct: {} \t pred_k: {} \t labels: {} \t correct_k: {} \t precision_k: {} \t AP: {}'.format(k, total_correct, pred_k[i], ql[i], correct_k, round(precision_k,3), round(AP,3)))

        # embed()
        # print()
        # print()
        # add AP to calculate mAP on all queries
        mAP += AP 

    mAP = (mAP / ql.size(0)) 
    # print(pred_k.shape)
    print('mAP@{}: {:.5f}'.format(n, mAP))
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


def visualize_rank_k(scores, features, output_dir, width=128, height=256, topk=10):
    qf = features["qf"]
    ql = features["ql"]
    gf = features["gf"]
    gl = features["gl"]

    query_paths = features['query_paths']
    gallery_paths = features['gallery_paths']

    mkdir_if_missing(output_dir)

    # return vector of index in scores metric that have highest score for each query: len = query frame
    indices = scores.topk(topk, dim=1)[1]

    # top k pred for each query, return matrix of (number query x n)
    pred = gl[indices]

    for i in range(ql.size(0)): 
        qimg = cv2.imread(query_paths[i])
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
        pid = ql[i].item()
        mkdir_if_missing(os.path.join(output_dir, str(pid)))

        # list of matched predicts (boolean tensor)
        matches = pred[i].eq(ql[i])

        rank_idx = 1

        for j in range(len(matches)):
            border_color = GREEN if matches[j] else RED
            gimg = cv2.imread(gallery_paths[indices[i][j].item()])
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

    
        imname = os.path.basename(query_paths[i])[:-4]
        cv2.imwrite(os.path.join(output_dir, str(pid), imname + '.jpg'), grid_img)
        print('Writing {}.jpg'.format(imname))

    print('Successfully.')


def evaluate(features, p_k, mAP_n, range, show = False, visualize_topk=5, infer_dir = None):
    qf = features["qf"]
    ql = features["ql"]
    gf = features["gf"]
    gl = features["gl"]
    
    query_paths = features['query_paths']
    gallery_paths = features['gallery_paths']

    # matrix of confidence with shape (query frames x gallery frames)
    scores = qf.mm(gf.t())

    acc_top1 = calculate_rank_1(scores, features)
    precision_k = calculate_precision_k(scores, features, k = p_k)
    mAP = calculate_mAP_n(scores, features, n = mAP_n)
    
    if show:
        visualize_rank_k(scores, features, output_dir=infer_dir, topk=visualize_topk)

    return acc_top1, precision_k, mAP




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--predict_path", default='predicts/features_new.pth', type=str)
    parser.add_argument("--p_k", default=5, type=int)
    parser.add_argument("--mAP_n", default=5, type=int)
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--visualize_top_k", default=10, type=int)
    parser.add_argument("--inference_dir", default = "inference_test", type=str)

    args = parser.parse_args()

    features = torch.load(args.predict_path)

    evaluate(features = features,
            p_k = args.p_k,
            mAP_n = args.mAP_n,
            range = 0,
            show = args.show,
            visualize_topk = args.visualize_top_k,
            infer_dir = args.inference_dir)

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

    # # matrix of confidence with shape (query frames x gallery frames)
    # scores = qf.mm(gf.t())

    # calculate_rank_1(scores)
    # calculate_precision_k(scores, args.p_k)
    # calculate_mAP_n(scores, args.mAP_n)
    # if args.show:
    #     visualize_rank_k(scores, args.inference_dir, topk=args.visualize_rank_k)



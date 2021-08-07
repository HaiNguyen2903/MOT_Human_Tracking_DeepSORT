import torch
from IPython import embed
import argparse

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--predict_path", default='predicts/features_train25epochs.pth', type=str)
parser.add_argument("--p_k", default=5, type=int)
parser.add_argument("--map_n", default=5, type=int)
args = parser.parse_args()


features = torch.load(args.predict_path)

'''
gf: querry frames: shape (frames x features_len) (208 x 512)
ql: querry labels: vector len = number of querry images
gf: gallery frames: shape (frames x features_len) (21549 x 512)
gl: gallery labels: vector len = number of gallery images
'''

qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

# matrix of confidence with shape (querry frames x gallery frames)
scores = qf.mm(gf.t())

# # return vector of index in scores metric that have highest score for each querry: len = querry frame
# res = scores.topk(5, dim=1)[1][:, 0]            

# # total equal values between ql and ql vectors
# top1correct = gl[res].eq(ql).sum().item()

# print("Accuracy top 1: {:.3f}".format(top1correct / ql.size(0)))

def calculate_rank_1(scores):
    res = scores.topk(5, dim=1)[1][:, 0]
    top1correct = gl[res].eq(ql).sum().item()
    print("Accuracy top 1: {:.3f}".format(top1correct / ql.size(0)))
    return top1correct / ql.size(0)


def calculate_precision_k(scores, k=5):
    # return vector of index in scores metric that have highest score for each querry: len = querry frame
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
    avg_acc /= ql.size(0)

    print('P@{}: {:.3f}'.format(k, avg_acc))
    return avg_acc


'''
Calculate precison@k and AP@n for each query instead of average on all queries
'''

def calculate_mAP_n(scores, n=5):
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

        # add AP to calculate mAP on all queries
        mAP += AP 

    mAP /= ql.size(0)

    print('mAP@{}: {:.3f}'.format(n, mAP))
    return mAP


if __name__ == '__main__':
    calculate_rank_1(scores)
    calculate_precision_k(scores, args.p_k)
    calculate_mAP_n(scores, args.map_n)


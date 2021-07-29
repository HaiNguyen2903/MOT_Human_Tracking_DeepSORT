import torch
from IPython import embed
import argparse

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--predict-path", default='predicts/features_train25epochs.pth', type=str)

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

# return vector of index in scores metric that have highest score for each querry: len = querry frame
res = scores.topk(5, dim=1)[1][:, 0]            

# total equal values between ql and ql vectors
top1correct = gl[res].eq(ql).sum().item()

print("Accuracy top 1: {:.3f}".format(top1correct / ql.size(0)))

embed()
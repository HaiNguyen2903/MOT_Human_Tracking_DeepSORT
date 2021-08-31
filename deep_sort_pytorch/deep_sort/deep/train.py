import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from IPython import embed
from model import Net
from custom_dataloader import *
import wandb

parser = argparse.ArgumentParser(description="Train on market1501")
# parser.add_argument("--data-dir", default='data', type=str)
# parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/reid_dataset/uet_reid', type=str)
parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
# parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--ckpt', default = './checkpoint/ckpt.t7', type=str)
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--batch', default=16, type=int)
parser.add_argument('--save-ckpt-path', default='checkpoint/debug.t7', type=str)
parser.add_argument('--save-result', default='training_curves/train.jpg', type=str)
parser.add_argument('--project-name', default='Reid_Deepsort', type=str)
parser.add_argument('--run-name', default='new_run', type=str)

args = parser.parse_args()

# device
# device = "cuda:{}".format(
#     args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"

device = torch.device(1)


if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
# test_dir = os.path.join(root, "test")

# test on gallery folder
test_dir = os.path.join(root, "gallery")

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4, pad_if_needed=True),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# trainloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
#     batch_size=args.batch, shuffle=True
# )

# testloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
#     batch_size=args.batch, shuffle=True
# )

trainloader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(train_dir, transform=transform_train),
    batch_size=args.batch, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(test_dir, transform=transform_test),
    batch_size=args.batch, shuffle=True
)

num_classes = max(len(trainloader.dataset.classes),
                  len(testloader.dataset.classes))

print("Num class:", num_classes)
# net definition

'''
Creating gallery labels for custom test after each epoch
================================================
'''
# gallery_features = torch.tensor([]).float()
# gallery_labels = torch.tensor([]).long()
# gallery_paths = []

# train_labels = torch.tensor([]).long()
# train_paths = []

# with torch.no_grad():
    
#     # embed(header='query')

#     for idx, (inputs, labels, path) in enumerate(testloader):
#         gallery_labels = torch.cat((gallery_labels, labels))
#         gallery_paths.extend(path)

#     for idx, (inputs, labels, path) in enumerate(trainloader):
#         train_labels = torch.cat((train_labels, labels))
#         train_paths.extend(path)

# embed(header='debug testloader')

# # tensor([41,  2,  1,  2, 39, 23, 22, 30,  1,  2])
# '''
# ['/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/47/637.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/11/837.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/10/334.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/11/863.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/45/528.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/30/278.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/3/61.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/37/1102.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/10/1011.jpg',
#  '/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset/train/11/630.jpg']
# '''


'''
=================================================
'''

start_epoch = 0
net = Net(num_classes=num_classes)

if args.resume:
    assert os.path.isfile(
        args.ckpt), "Error: no checkpoint file found!"
    print('Loading from {}'.format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location=device)
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in net_dict.items() if k not in ["classifier.4.weight", "classifier.4.bias"]}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

best_acc = 0.

# train function for each epoch


def train(epoch):
    print("\nEpoch : %d" % (epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels, paths) in enumerate(trainloader):
        # embed(header = 'debug label 1')
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        # batch_size x num_class
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        # embed(header = 'debug train epoch')

        # backward
        # try:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # except:
        #     embed(header='debug label 2')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()

        # res = scores.topk(5, dim=1)[1][:, 0]
        # top1correct = gl[res].eq(ql).sum().item()

        total += labels.size(0)

        # print
        if (idx+1) % interval == 0:
            end = time.time()
            print("Epoch: {} \t [progress: {:.1f}%] \t time: {:.2f}s \t Loss:{:.5f} \t Correct:{}/{} \t Acc:{:.3f}%".format(epoch+1,
                100.*(idx+1)/len(trainloader), end-start, training_loss /
                interval, correct, total, 100.*correct/total
            ))

            training_loss = 0.
            start = time.time()

    print('Accuracy overall for epoch {}: {:.5f}'.format(epoch, correct/total))
    print('Total corrects for epoch {}: {}'.format(epoch, correct))
    print('Total samples for epoch {}: {}'.format(epoch, total))
    print()
    return train_loss/len(trainloader), 100.*correct/total


def test(epoch):
    global best_acc

    global net

    net_eval = net(reid=True)   # change net

    net_eval.eval()     # change net

    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()

    with torch.no_grad():
        print("Testing ...")
        # embed(header='debug each test batch')

        for idx, (inputs, labels, paths) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_eval(inputs)  # change net
            loss = criterion(outputs, labels)

            embed(header='debug inside test batch')

            test_loss += loss.item()

            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

            # correct += outputs.max(dim=1)[1].eq(gallery_labels).sum().item()

        end = time.time()
        print("Epoch: {} \t [progress: {:.1f}%] \t time: {:.2f}s \t Loss:{:.5f} \t Correct:{}/{} \t Acc:{:.3f}%".format(epoch+1,
            100.*(idx+1)/len(testloader), end-start, test_loss /
            len(testloader), correct, total, 100.*correct/total
        ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to {}".format(args.save_ckpt_path))
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, args.save_ckpt_path)

    return test_loss/len(testloader), acc


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="Loss")
ax1 = fig.add_subplot(122, title="Top 1 accuracy")


def draw_curve(epoch, train_loss, train_acc, test_loss, test_acc):
    global record
    record['train_loss'].append(train_loss)
    record['train_acc'].append(train_acc)
    record['test_loss'].append(test_loss)
    record['test_acc'].append(test_acc)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_acc'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_acc'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(args.save_result)

# lr decay


def lr_decay():
    global optimizer
    # embed(header='debug optimizer')
    for params in optimizer.param_groups:
        # params['lr'] *= 0.1
        # reduce by 2 
        params['lr'] /= 2 
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    # run = wandb.init(project = args.project_name, tags=["training vtx"])
    # run.name = args.run_name

    global optimizer

    for epoch in range(start_epoch, start_epoch + args.epochs):

        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        draw_curve(epoch, train_loss, train_acc, test_loss, test_acc)

        # wandb.log({'train/loss': train_loss,
        #             'train/top1_acc': train_acc}
        #             )

        # for params in optimizer.param_groups:
        #     wandb.log({'train/lr': params['lr']})

        # wandb.log({'test/loss': test_loss,
        #             'test/top1_acc': test_acc}
        #             )

        # if (epoch+1) % 20 == 0:
        # decay for every 5 epochs
        if (epoch+1) % 5 == 0:
            lr_decay()


if __name__ == '__main__':
    main()
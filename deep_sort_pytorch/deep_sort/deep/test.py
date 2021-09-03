# from deep_sort_pytorch.deep_sort.deep.custom_dataloader import ImageFolderWithPaths
import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from torchvision.datasets.folder import default_loader
from traitlets.traitlets import default

from custom_dataloader import ImageFolderWithPaths

from model import Net
from IPython import embed


# parser = argparse.ArgumentParser(description="Train on market1501")
# parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/reid_dataset/uet_reid', type=str)
# # parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset', type=str)
# parser.add_argument("--no-cuda", action="store_true")
# parser.add_argument("--gpu-id", default=1, type=int)
# parser.add_argument("--ckpt", default="./checkpoint/ckpt.t7", type=str)
# parser.add_argument("--batch", default=16, type=int)
# parser.add_argument("--save-path", default="predicts/debug.pth", type=str)

# args = parser.parse_args()

# device
# device = "cuda:{}".format(
#     args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"

# device = torch.device(1)
# # device = torch.device("cuda:1")

# if torch.cuda.is_available() and not args.no_cuda:
#     cudnn.benchmark = True

# # data loader
# root = args.data_dir
# query_dir = os.path.join(root, "query")
# gallery_dir = os.path.join(root, "gallery")

# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((128, 64)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


# queryloader = torch.utils.data.DataLoader(
#     ImageFolderWithPaths(query_dir, transform=transform),
#     batch_size=args.batch, shuffle=False
# )
# '''
# try load gallery random
# '''
# galleryloader = torch.utils.data.DataLoader(
#     ImageFolderWithPaths(gallery_dir, transform=transform),
#     batch_size=args.batch, shuffle=True
# )

# # embed()

# # net definition
# net = Net(reid=True)


# assert os.path.isfile(
#     args.ckpt), "Error: no checkpoint file found!"
# print('Loading from {}'.format(args.ckpt))

# checkpoint = torch.load(args.ckpt, map_location=device)

# net_dict = checkpoint['net_dict']

# net.load_state_dict(net_dict, strict=False)    

# net.eval()

# '''
# For multiple gpu
# '''

# # net.to(device)

# net.to(device)

# # compute features
# query_features = torch.tensor([]).float()
# query_labels = torch.tensor([]).long()
# query_paths = []

# gallery_features = torch.tensor([]).float()
# gallery_labels = torch.tensor([]).long()
# gallery_paths = []


# with torch.no_grad():
#     for idx, (inputs, labels, paths) in enumerate(queryloader):        

#         inputs = inputs.to(device)

#         features = net(inputs).cpu()

#         query_features = torch.cat((query_features, features), dim=0)
#         query_labels = torch.cat((query_labels, labels))
#         query_paths.extend(paths)

#         embed(header='debug model')

#     for idx, (inputs, labels, paths) in enumerate(galleryloader):
#         inputs = inputs.to(device)
#         features = net(inputs).cpu()

#         gallery_features = torch.cat((gallery_features, features), dim=0)
#         gallery_labels = torch.cat((gallery_labels, labels))
#         gallery_paths.extend(paths)

#         # embed(header='gallery')

# # gallery_labels -= 2

# # save features
# features = {
#     "qf": query_features,
#     "ql": query_labels,
#     "gf": gallery_features,
#     "gl": gallery_labels,
#     "query_paths": query_paths,
#     "gallery_paths": gallery_paths
# }

# torch.save(features, args.save_path)


def calculate_features(net, root_dir, batch, device, save_features = True, save_path = None):
    net.eval()
    net.to(device)

    query_dir = os.path.join(root_dir, "query")
    gallery_dir = os.path.join(root_dir, "gallery")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    queryloader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(query_dir, transform=transform),
        batch_size=batch, shuffle=False
        )

    '''
    try load gallery random
    '''
    galleryloader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(gallery_dir, transform=transform),
        batch_size=batch, shuffle=True
        )

    # compute features
    query_features = torch.tensor([]).float()
    query_labels = torch.tensor([]).long()
    query_paths = []

    gallery_features = torch.tensor([]).float()
    gallery_labels = torch.tensor([]).long()
    gallery_paths = []

    with torch.no_grad():
        print('Calculating embedding for query ...')
        for idx, (inputs, labels, paths) in enumerate(queryloader):        

            inputs = inputs.to(device)

            features = net(inputs).cpu()

            query_features = torch.cat((query_features, features), dim=0)
            query_labels = torch.cat((query_labels, labels))
            query_paths.extend(paths)

            # embed(header='debug model')


        print('Calculating embedding for gallery ...')
        for idx, (inputs, labels, paths) in enumerate(galleryloader):
            inputs = inputs.to(device)
            features = net(inputs).cpu()

            gallery_features = torch.cat((gallery_features, features), dim=0)
            gallery_labels = torch.cat((gallery_labels, labels))
            gallery_paths.extend(paths)

    # save features
    features = {
        "qf": query_features,
        "ql": query_labels,
        "gf": gallery_features,
        "gl": gallery_labels,
        "query_paths": query_paths,
        "gallery_paths": gallery_paths
    }

    if save_features:
        torch.save(features, save_path)
        print('Save features to {}'.format(save_path))

    return features
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/reid_dataset/uet_reid', type=str)
    # parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset', type=str)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", default=1, type=int)
    parser.add_argument("--ckpt", default="./checkpoint/ckpt.t7", type=str)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--save-path", default="predicts/debug.pth", type=str)
    parser.add_argument("--device", default=1, type=int)

    args = parser.parse_args()

    device = torch.device(args.device)

    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True

    # define net 
    net = Net(reid=True)

    assert os.path.isfile(
    args.ckpt), "Error: no checkpoint file found!"

    print('Loading from {}'.format(args.ckpt))

    checkpoint = torch.load(args.ckpt, map_location=device)

    net_dict = checkpoint['net_dict']

    net.load_state_dict(net_dict, strict=False)    

    calculate_features(net = net, root_dir = args.data_dir, batch = args.batch, device=device, save_path=args.save_path)




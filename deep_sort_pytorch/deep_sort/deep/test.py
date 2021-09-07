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
import time

def calculate_features(net, root_dir, batch, device, save_features = True, save_dir = None, save_name = None):
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
            print('Loading query batch {}'.format(idx))   

            inputs = inputs.to(device)

            features = net(inputs).cpu()

            query_features = torch.cat((query_features, features), dim=0)
            query_labels = torch.cat((query_labels, labels))
            query_paths.extend(paths)

            # embed(header='debug model')


        print('Calculating embedding for gallery ...')
        for idx, (inputs, labels, paths) in enumerate(galleryloader):
            inputs = inputs.to(device)
            start = time.time()
            features = net(inputs).cpu()
            end = time.time()
            print('Time for gallery batch {}: {}'.format(idx, end-start))

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
        # save_name = os.path.basename(args.ckpt)[:-3]
        save_path = os.path.join(save_dir, 'features_' + save_name + '.pth')
        torch.save(features, save_path)
        print('Save features to {}'.format(save_path))

    return features
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/DATA_ROOT/combine_dataset/reid_dataset', type=str)
    # parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/reid_dataset/uet_reid', type=str)
    # parser.add_argument("--data-dir", default='/data.local/hangd/data_vtx/toy_data/toy_reid_dataset/reid_dataset', type=str)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", default=1, type=int)
    parser.add_argument("--ckpt", default="./checkpoint/ckpt.t7", type=str)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--save-path", default="predicts/debug.pth", type=str)
    parser.add_argument("--save-dir", default="predicts/", type=str)
    parser.add_argument("--save-name", default="debug.pth", type=str)
    parser.add_argument("--device", default=1, type=int)

    args = parser.parse_args()

    device = torch.device(args.device)

    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True

    # data loading
    root = args.data_dir
    train_dir = os.path.join(root, "train")
    # test_dir = os.path.join(root, "test")

    # test on gallery folder
    test_dir = os.path.join(root, "gallery")

    trainloader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(train_dir, transform=None),
            batch_size=args.batch, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(test_dir, transform=None),
        batch_size=args.batch, shuffle=True
    )

    num_classes = max(len(trainloader.dataset.classes),
                    len(testloader.dataset.classes))



    print("Num class:", num_classes)

    # to test on old model
    # num_classes=868

    # define net 
    net = Net(reid=True, num_classes=num_classes)


    assert os.path.isfile(
    args.ckpt), "Error: no checkpoint file found!"

    print('Loading from {}'.format(args.ckpt))

    checkpoint = torch.load(args.ckpt, map_location=device)

    net_dict = checkpoint['net_dict']

    net.load_state_dict(net_dict, strict=False)    

    calculate_features(net = net, root_dir = args.data_dir, batch = args.batch, device=device, save_dir=args.save_dir, save_name=args.save_name)




import torch
from torchvision import datasets
from IPython import embed
import os

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 

        # tuple of (image, label) (wrong label, expected label as folder name)
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # get path to dir contains images and extract dir name as label
        dir_path = os.path.dirname(path)    
        label = int(os.path.basename(dir_path)) - 1 

        # make a new tuple that includes original and the path
        tuple_with_path = ((original_tuple[0], label) + (path,))
        
        # tuple_with_path = (original_tuple + (path,))
        # embed(header = 'debug dataloader')
        return tuple_with_path

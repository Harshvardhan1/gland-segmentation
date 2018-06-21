import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import random
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class glandDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='dataset/'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

    def __len__(self):
        return 145

    def transforms(self,image,mask):

        #image = TF.to_pil_image(image)
        #mask  = TF.to_pil_image(mask)

        image = image.resize((775,522))
        mask = mask.resize((775,522))
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random crop
        #print np.shape(image)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(508, 508))
        #print i,j,h,w
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        

        # Since Unet is used, reduce target size
        mask = TF.resize(mask,244)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


    def __getitem__(self, idx):
        if(idx<85):
            img_name = self.root_dir+'train_'+str(idx+1)+'.bmp'
            mask_name = self.root_dir+'train_'+str(idx+1)+'_anno.bmp'
        else:
            img_name = self.root_dir+'testA_'+str(idx-84)+'.bmp'
            mask_name = self.root_dir+'testA_'+str(idx-84)+'_anno.bmp'
        
        image = Image.open(img_name)
        mask = Image.open(mask_name)

        x,y = self.transforms(image,mask)
        
        return x,y

class glandTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='dataset/'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        img_name = self.root_dir+'testB_'+str(idx+1)+'.bmp'
        mask_name = self.root_dir+'testB_'+str(idx+1)+'_anno.bmp'
        
        image = Image.open(img_name)
        mask = Image.open(mask_name)

        x = image.resize((508,508))
        y = mask.resize((244,244))

        return TF.to_tensor(x),TF.to_tensor(y)

'''
def random_shift_scale_rotate(image, angle, scale, aspect, shift_dx, shift_dy,
                              borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if len(image.shape) == 3:  # Img or mask
        height, width, channels = image.shape
    else:
        height, width = image.shape

    sx = scale * aspect / (aspect ** 0.5)
    sy = scale / (aspect ** 0.5)
    dx = round(shift_dx * width)
    dy = round(shift_dy * height)

    cc = np.math.cos(angle / 180 * np.math.pi) * sx
    ss = np.math.sin(angle / 180 * np.math.pi) * sy
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(0, 0, 0, 0))
    return image
'''
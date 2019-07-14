# -*- coding: utf-8 -*-


from torch.utils.data import Dataset
import glob
from PIL import Image
import random

class ImageDataset(Dataset):
    def __init__(self, root, transforms_, unaligned=False, mode="train"):
        self.transforms = transforms_
        self.unaligned = unaligned
        self.file_X = sorted(glob.glob(f'{root}/{mode}/X' + '/*.*'))
        self.file_Y = sorted(glob.glob(f'{root}/{mode}/Y' + '/*.*'))
        
    def __getitem__(self, index):
        img_X = Image.open(self.file_X[index % len(self.file_X)]).convert('RGB')
        if self.unaligned:
            img_Y = Image.open(self.file_Y[random.randint(0, len(self.file_Y)-1)]).convert('RGB')
        else:
            img_Y = Image.open(self.file_Y[index % len(self.file_Y)]).convert('RGB')
        
        img_X = self.transforms(img_X)
        img_Y = self.transforms(img_Y)
        
        return {'X': img_X, 'Y': img_Y}
    
    def __len__(self):
        return max(len(self.file_X), len(self.file_Y))
            
        
    
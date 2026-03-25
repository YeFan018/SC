# Data_Loader.py: 加载文本和图像特征
import torch
import numpy as np
from torch.utils.data import Dataset

class LoadData(Dataset):
    def __init__(self, filename):
        self.name = filename
        t_file = r'C:\path\to\text\feature_extracted_text.npz'
        i_file = r'C:\path\to\image\feature_extracted_img.npz'

        text = np.load(t_file)
        self.data_text = torch.from_numpy(text['data']).float()

        img = np.load(i_file)
        self.data_img = torch.from_numpy(img['data']).squeeze().float()

    def __len__(self):
        return self.data_text.shape[0]

    def __getitem__(self, item):
        return self.data_text[item], self.data_img[item]

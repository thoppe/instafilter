import cv2

import torch
from torch.utils.data import Dataset
from instafilter.utils import features_from_image


class ColorizedDataset(Dataset):
    def __init__(self, f_source, f_target, device):

        img0 = cv2.imread(str(f_source))
        f0 = features_from_image(img0)

        img1 = cv2.imread(str(f_target))
        f1 = features_from_image(img1)

        assert f0.shape == f1.shape

        self.x = torch.tensor(f0).to(device)
        self.y = torch.tensor(f1).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

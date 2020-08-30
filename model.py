import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
from utils import features_from_image


class ColorNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 5)

        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        return x


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

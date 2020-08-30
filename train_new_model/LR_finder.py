import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from model import ColorNet, ColorizedDataset

import pylab as plt

device = "cuda"
batch_size = 2 ** 10
# learning_rate = 0.01

loss_func = torch.optim.Adam
loss_func = torch.optim.AdamW


f_source = "samples/Normal.jpg"
f_target = "samples/Earlybird.jpg"
data = ColorizedDataset(f_source, f_target, device=device)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)


LR = np.logspace(-5, 1, 200)
random_seed = 124

df_data = []


torch.manual_seed(random_seed)

net = ColorNet()
criterion = nn.L1Loss()
optimizer = loss_func(net.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

net.to(device)
net.train()  # prep model for training

avg_loss = 0.0
beta = 0.98
batch_num = 0

for learning_rate, (data, target) in zip(tqdm(LR), train_loader):
    batch_num += 1

    optimizer.param_groups[0]["lr"] = learning_rate

    optimizer.zero_grad()

    output = net(data)
    loss = criterion(output, target)

    batch_loss = loss.data.cpu().numpy()
    avg_loss = beta * avg_loss + (1 - beta) * batch_loss
    smoothed_loss = avg_loss / (1 - beta ** batch_num)

    loss.backward()
    optimizer.step()

    df_data.append({"learning_rate": learning_rate, "loss": smoothed_loss})


df = pd.DataFrame(df_data)

plt.plot(df.learning_rate, df.loss)
plt.xscale("log", base=10)
plt.show()

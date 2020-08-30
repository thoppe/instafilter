from model import ColorNet
import numpy as np
from utils import features_from_image
import cv2
import torch
from tqdm import tqdm

device = "cpu"
device = "cuda"

f_source = "samples/Normal.jpg"
f_model = "models/Earlybird.pt"

img = cv2.imread(str(f_source))

# Expect an image in BGR format
torch.set_grad_enabled(False)

state = torch.load(f_model)
net = ColorNet()
net.load_state_dict(state)
net.eval()
net.to(device)

f0 = features_from_image(img)

for _ in tqdm(range(2000)):
    f0 = torch.tensor(f0).to(device)

    f1 = net(f0)
    f1 = f1.clone().detach().cpu().numpy()

    bgr = np.clip(f1[:, :3] * 255, 0, 255).astype(np.uint8)
    bgr = bgr.reshape(img.shape)

cv2.imwrite("example.jpg", bgr)

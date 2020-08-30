import numpy as np
import torch
from pathlib import Path

from .utils import features_from_image
from .model import ColorNet

model_location = Path(__file__).resolve().parent / "models"


class Instafilter:
    def __init__(self, name, device="cuda"):
        self.device = device
        self.net = self.load_model(name)

    @staticmethod
    def get_models():
        return [x.stem for x in model_location.glob("*.pt")]

    def load_model(self, name):
        f_model = model_location / (name + ".pt")

        if not f_model.exists():
            names = ' '.join(sorted(self.get_models()))
            raise KeyError(f"Model {name} not found. Known models: {names}")

        state = torch.load(f_model)

        net = ColorNet()
        net.load_state_dict(state)
        net.eval()
        net.to(self.device)

        return net

    def __call__(self, img):
        """
        Filters the image with the loaded model. Input img is expected to
        be in BGR with the dimensions (height, width, channels=3).

        Returns an image in the same format and shape.
        """

        f0 = features_from_image(img)
        f0 = torch.tensor(f0).to(self.device)
        f1 = self.net(f0)
        f1 = f1.clone().detach().cpu().numpy()

        bgr = np.clip(f1[:, :3] * 255, 0, 255).astype(np.uint8)
        bgr = bgr.reshape(img.shape)

        return bgr

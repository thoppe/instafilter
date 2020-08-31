import numpy as np
import torch
from pathlib import Path

from .utils import features_from_image
from .model import ColorNet

model_location = Path(__file__).resolve().parent / "models"


class Instafilter:
    def __init__(self, name, device="cuda", batch_size=2 ** 16):
        """
        Load an instafilter for processing.
        name must be one from Instafilter.get_models()
        device should be one of "cuda" (GPU) or "cpu".
        """

        self.device = device
        self.net = self.load_model(name)
        self.batch_size = batch_size

    @staticmethod
    def get_models():
        return [x.stem for x in model_location.glob("*.pt")]

    def load_model(self, name):
        f_model = model_location / (name + ".pt")

        if not f_model.exists():
            names = " ".join(sorted(self.get_models()))
            raise KeyError(f"Model {name} not found. Known models: {names}")

        state = torch.load(f_model)

        net = ColorNet()
        net.load_state_dict(state)
        net.eval()
        net.to(self.device)

        return net

    @staticmethod
    def _load_image_from_filename(f_image):
        """
        Loads an image from a filename. Raises an error if the file
        does not exist. Uses CV2.
        """

        try:
            import cv2
        except ModuleNotFoundError:  # pragma: no cover

            raise ImportError(
                "cv2 required to load images from a file, run\n"
                "pip install opencv-python"
            )

        if not Path(f_image).exists():
            raise OSError(f_image)

        return cv2.imread(str(f_image))

    def __call__(self, img, is_RGB=False):
        """
        Filters the image with the loaded model. Input img is expected to
        be in BGR with the dimensions (height, width, channels).

        Returns an image in the same format and shape.
        """

        if isinstance(img, (str, Path)):
            img = self._load_image_from_filename(img)

        # Remove alpha channel if it exists
        img = img[:, :, :3]

        if is_RGB:
            img = img[:, :, ::-1]

        f0 = features_from_image(img)

        f1 = []
        for ndx in range(0, len(f0), self.batch_size):
            batch = f0[ndx : min(ndx + self.batch_size, len(f0))]
            batch = torch.tensor(batch).to(self.device)
            result = self.net(batch)
            result = result.clone().detach().cpu().numpy()
            f1.append(result)

        f1 = np.vstack(f1)

        bgr = np.clip(f1[:, :3] * 255, 0, 255).astype(np.uint8)
        bgr = bgr.reshape(img.shape)

        if is_RGB:
            img = img[:, :, ::-1]

        return bgr

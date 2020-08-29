from model import ColorNet
import numpy as np
from utils import features_from_image
import torch
from pathlib import Path

model_location = Path('models')

class Instafilter:

    def __init__(self, name=None, device='cuda'):
        self.device = device
        self.models = {}        
        self.current_model = None

        if name is not None:
            self.load_model(name)

    def get_models(self):
        names = sorted([x.stem for x in model_location.glob('*.pt')])
        return ' '.join(names)

    def load_model(self, name):

        self.current_model = name
        
        if name in self.models:
            return True
        
        f_model = model_location / (name + '.pt')

        if not f_model.exists():
            raise KeyError(
                f"Model {name} not found. Known models: {self.get_models()}")

        state = torch.load(f_model)
        
        net = ColorNet()
        net.load_state_dict(state)
        net.eval()
        net.to(self.device)

        self.models[name] = net

    def __call__(self, img):
        """
        Runs the current loaded insta filter against the image.
        Expects the image to be in (height, width, channels=3) and BGR.
        """

        net = self.models[self.current_model] 
        
        f0 = features_from_image(img)
        f0 = torch.tensor(f0).to(self.device)
        f1 = net(f0)
        f1 = f1.clone().detach().cpu().numpy()
            
        bgr = np.clip(f1[:,:3] * 255, 0, 255).astype(np.uint8)
        bgr = bgr.reshape(img.shape)

        return bgr      

if __name__ == "__main__":
    import cv2
    
    F = Instafilter("Nashville")
    img1 = cv2.imread('samples/Normal.jpg')
    img2 = F(img1)
    cv2.imwrite("example2.jpg", img2)
        


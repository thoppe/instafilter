import cv2
from pathlib import Path
from instafilter import Instafilter

scale_size = 0.741

f_source = "train_new_model/input/Normal.jpg"
save_dest = Path("examples")
save_dest.mkdir(exist_ok=True)

img0 = cv2.imread(f_source)

for name in Instafilter.get_models():

    f_save = save_dest / (name + ".jpg")
    model = Instafilter(name)

    img1 = model(f_source)
    img2 = cv2.resize(img1, None, fx=scale_size, fy=scale_size)

    print(f"Saving {name}, {img2.shape}")
    cv2.imwrite(str(f_save), img2)

import instafilter
import cv2
from pathlib import Path

F = instafilter.Instafilter()
names = F.get_models().split()

f_source = 'samples/Normal.jpg'
save_dest = Path('examples')
save_dest.mkdir(exist_ok=True)

img0 = cv2.imread(f_source)

for name in names:
    print(f"Saving {name}")
    f_save = save_dest / (name + '.jpg')
    f = instafilter.Instafilter(name)
    img1 = f(img0)

    cv2.imwrite(f_save, img1)
    
    

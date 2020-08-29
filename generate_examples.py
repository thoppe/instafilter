import instafilter

import cv2
from pathlib import Path

scale_size = 0.5

F = instafilter.Instafilter()
names = F.get_models().split()

f_source = 'training_samples/Normal.jpg'
save_dest = Path('samples')
save_dest.mkdir(exist_ok=True)

img0 = cv2.imread(f_source)

'''
for name in names:

    f_save = save_dest / (name + '.jpg')
    f = instafilter.Instafilter(name)

    img1 = f(img0)
    img2 = cv2.resize(img1, None, fx=scale_size, fy=scale_size)

    print(f"Saving {name}, {img2.shape}")
    cv2.imwrite(str(f_save), img2)
'''

import pylab as plt

fig, axes = plt.subplots(9, 5, figsize=(8,11))
#fig, axes = plt.subplots(3, 1, figsize=(8,11))

axes = axes.ravel()

images = []
for name, ax in zip(names[:], axes):
    f_save = save_dest / (name + '.jpg')
    img = plt.imread(f_save) 
    ax.imshow(img, interpolation='nearest')
    ax.set_title(name, color='w')

for ax in axes:
    ax.axis('off')
    ax.axis("image")

plt.tight_layout(h_pad=0, w_pad=0, pad=0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

plt.show()


    
    
    

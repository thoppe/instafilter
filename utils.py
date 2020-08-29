import numpy as np


def features_from_image(img):

    # Blue, Green, & Red channels
    BGR = img.astype(np.float32) / 255.0

    # Saturation and lightness
    mx, mn = BGR.max(axis=2), BGR.min(axis=2)
    avg = BGR.mean(axis=2)

    LS = np.array([avg, mx - mn]).transpose(1, 2, 0)

    features = np.dstack([BGR, LS])
    h, w, nf = features.shape

    return features.reshape(h * w, nf)

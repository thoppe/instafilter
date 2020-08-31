import pytest

from pathlib import Path

from instafilter import Instafilter
import cv2

__local__ = Path(__file__).resolve().parent


def test_get_model_names():
    """
    Check that the module loads the list of names and is non-empty.
    """

    names = Instafilter.get_models()
    assert isinstance(names, list)
    assert len(names) > 1


def test_missing_model():
    """
    Try to load a filter that doesn't exists.
    Raise KeyError
    """

    with pytest.raises(KeyError):
        Instafilter("xxx")


def test_missing_image():
    """
    Try to filter an image that doesn't exist.
    Raise FileNotFoundError
    """

    model = Instafilter("Lo-Fi")

    with pytest.raises(OSError):
        model("xxx")


def test_filter_image():
    """
    Filter an image. Check that it changed.
    """

    model = Instafilter("Lo-Fi")

    f_image = __local__ / "Normal.jpg"

    img0 = cv2.imread(str(f_image))
    img1 = model(f_image)

    diff = (img0 - img1).sum()

    assert abs(diff) > 0


def test_RGB_mode():
    """
    Filter an image with RGB mode on and off.
    Check that we get different results.
    """

    model = Instafilter("Lo-Fi")

    f_image = __local__ / "Normal.jpg"

    img1 = model(f_image)
    img2 = model(f_image, is_RGB=True)

    diff = (img1 - img2).sum()

    assert abs(diff) > 0

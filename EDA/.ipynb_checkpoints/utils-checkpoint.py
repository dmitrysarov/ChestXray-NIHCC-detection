from PIL import Image


def get_size(fp):
    return Image.open(fp).size

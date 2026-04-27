from .loader import load_image


def main():
    img = load_image()
    print(img.size)

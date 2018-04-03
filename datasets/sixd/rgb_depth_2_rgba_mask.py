import sys
from pathlib import Path
import numpy as np
import cv2
from progressbar import ProgressBar


def gen_mask(path):
    files = path.glob('depth/*.png')
    for file in ProgressBar()(list(map(Path, files))):
        depth = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH)
        mask = np.greater(depth, 0) * 255
        cv2.imwrite(str(path / 'mask' / file.name), mask)


def gen_rgba(path):
    files = path.glob('rgb/*.png')
    for file in ProgressBar()(list(map(Path, files))):
        rgb = cv2.imread(str(file))
        mask = cv2.imread(str(path / 'mask' / file.name), cv2.IMREAD_GRAYSCALE)
        # erode and blur mask to avoid black and jagged edges
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # pre multiply alpha channel into rgb for correct interpolation in OpenGL
        rgb = rgb.astype(np.uint16) * mask.reshape((*mask.shape, 1))
        rgb = (rgb // 255).astype(np.uint8)
        # merge and save
        masked_img = cv2.merge((*cv2.split(rgb), mask))
        cv2.imwrite(str(path / 'rgba' / file.name), masked_img)


def main():
    if len(sys.argv) != 3:
        print("Use: python3 [path to this script] ['mask' or 'rgba'] [folder path]")
        return

    _, command, path = sys.argv

    path = Path(path)
    assert path.is_dir()

    if command == 'mask':
        gen_mask(path)
    elif command == 'rgba':
        gen_rgba(path)
    else:
        raise 1


if __name__ == '__main__':
    main()

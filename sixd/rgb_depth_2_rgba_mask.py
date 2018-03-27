import numpy as np
import cv2
import glob


def gen_masks():
    files = glob.glob('./depth/*.png')
    for i, file in enumerate(files):
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
        mask = np.greater(img, 0) * 255
        cv2.imwrite(file.replace('depth', 'mask'), mask)
        if i % 50 == 0:
            print(i / len(files))


def gen_rgba():
    files = glob.glob('./rgb/*.png')
    B = np.ones((3, 3), dtype='uint8')
    gauss3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    for i, file in enumerate(files):
        img = cv2.imread(file)
        mask = cv2.imread(file.replace('rgb', 'mask'), 0)
        mask = cv2.erode(mask, B)
        mask = cv2.filter2D(mask, -1, gauss3x3)
        b, g, r = cv2.split(img)
        rgba = cv2.merge((b, g, r, mask))

        cv2.imwrite(file.replace('rgb', 'rgba'), rgba)
        if i % 50 == 0:
            print(i / len(files))

import sys
import os
import cv2


def main():
    root_dir = sys.argv[1]

    inst_mask_dir = root_dir + '/inst_mask'
    box_mask_dir = root_dir + '/box_mask'
    assert os.path.isdir(inst_mask_dir)
    assert os.path.isdir(box_mask_dir)

    out_dir = root_dir + '/mask'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    file_names = list(filter(
        lambda f: os.path.splitext(f)[-1],
        os.listdir(box_mask_dir)
    ))

    for file_name in sorted(file_names):
        inst_mask = cv2.imread(inst_mask_dir + '/' + file_name, cv2.IMREAD_UNCHANGED)
        box_mask = cv2.imread(box_mask_dir + '/' + file_name, cv2.IMREAD_GRAYSCALE)
        mask = inst_mask * (box_mask // 255)
        cv2.imwrite(out_dir + '/' + file_name, mask)


if __name__ == '__main__':
    main()

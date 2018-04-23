import numpy as np
from pathlib import Path
from Mask_RCNN import utils
from Mask_RCNN.config import Config
import cv2


class Overlay3DConfig(Config):
    NAME = "overlay3d"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 2  # background + 2 classes

    IMAGE_RESIZE_MODE = "none"  # 640x480

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 30


class Overlay3DDataset(utils.Dataset):
    def load_overlay3d(self, overlay3dOutDir):
        self.add_class("overlay3d", 1, "cup")
        self.add_class("overlay3d", 2, "carton")

        out_dir = Path(overlay3dOutDir)
        assert out_dir.exists()

        rgb_paths = out_dir.glob('rgb/*.png')
        for p in rgb_paths:
            id = p.name[:-4]
            self.add_image("overlay3d", image_id=id, path=str(p))

    def image_reference(self, image_id):
        """Return the image_def data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "overlay3d":
            return info["overlay3d"]
        else:
            super(self.__class__).image_reference(image_id)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img = cv2.imread(info['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        p = Path(info['path']).parent.parent / 'mask' / (info['id'] + '.png')
        p = str(p)
        mask = cv2.imread(p, cv2.IMREAD_UNCHANGED)

        masks = []
        class_ids = []

        for rng, class_id in [((40, 80), 1), ((80, 120), 2)]:
            for i in range(rng[0], rng[1]):
                m = mask == i
                area = m.sum()
                if area > 20 ** 2:
                    masks.append(m)
                    class_ids.append(class_id)

        masks = np.stack(masks, axis=-1)

        return masks, np.array(class_ids, dtype=np.int)


def main():
    from matplotlib import pyplot as plt

    dataset = Overlay3DDataset()
    dataset.load_overlay3d('../datasets/sixd/overlay_3D/out')

    img = dataset.load_image(0)
    mask, _ = dataset.load_mask(0)
    mask = np.argmax(mask, axis=2)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    main()

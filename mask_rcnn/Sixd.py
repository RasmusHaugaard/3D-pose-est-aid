import numpy as np
from progressbar import ProgressBar
import yaml

from pathlib import Path
from Mask_RCNN import utils
from Mask_RCNN.config import Config
from overlay_instances_2d import overlay_instances_cpu as overlay

DEFAULT_DATASET_YEAR = 2017


class SixdConfig(Config):
    NAME = "sixd"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 2  # background + 2 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 25


class SixdDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_sixd(self, count, coco_dataset_dir, coco_subset, sixd_train_dir, coco_year=DEFAULT_DATASET_YEAR):
        # Add classes
        self.add_class("sixd", 1, "cup")
        self.add_class("sixd", 2, "carton")

        def_path = Path("./image_def_{}{}_{}.yaml".format(coco_subset, coco_year, count))
        if def_path.is_file():
            print('Using cached definition file:', def_path)
            with def_path.open() as stream:
                image_defs = yaml.load(stream)
        else:
            print('Creating and caching definition file:', def_path)

            image_dir = "{}/{}{}".format(coco_dataset_dir, coco_subset, coco_year)
            all_images = list(Path(image_dir).glob('*.jpg'))
            image_paths = np.random.choice(all_images, count)

            cup_files = list((Path(sixd_train_dir) / '01' / 'rgba').glob('*.png'))
            carton_files = list((Path(sixd_train_dir) / '02' / 'rgba').glob('*.png'))

            image_defs = []
            for i, image_path in enumerate(ProgressBar()(image_paths)):
                instance_count = np.random.randint(5, 15)
                cup_count = round(instance_count * np.random.rand())
                carton_count = instance_count - cup_count

                cup_paths = [(1, path) for path in np.random.choice(cup_files, cup_count)]
                carton_paths = [(2, path) for path in np.random.choice(carton_files, carton_count)]
                instance_paths = cup_paths + carton_paths
                np.random.shuffle(instance_paths)
                class_ids = np.array([c for c, _ in instance_paths])
                instance_paths = [str(p) for _, p in instance_paths]

                bg_path = str(image_path)
                image_def = overlay.generate_image_def(bg_path, instance_paths, class_ids)
                image_defs.append(image_def)

            with def_path.open(mode='w', encoding='utf8') as stream:
                yaml.dump(image_defs, stream)

        for i, image_def in enumerate(image_defs):
            class_ids = [inst['class'] for inst in image_def['instances']]
            self.add_image(
                "sixd", image_id=i,
                path=image_def['background']['path'],
                image_def=image_def,
                class_ids=np.array(class_ids, np.int32),
            )

    def load_image(self, image_id):
        # Load image
        info = self.image_info[image_id]
        image_def = info['image_def']
        image = overlay.generate_image(image_def)
        return image

    def image_reference(self, image_id):
        """Return the image_def data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sixd":
            return info["sixd"]
        else:
            super(self.__class__).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for cups/cartons of the given image ID.
        """
        info = self.image_info[image_id]
        image_def, class_ids = info['image_def'], info['class_ids']
        masks = overlay.generate_masks(image_def)
        return masks, class_ids

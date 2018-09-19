import numpy as np
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn.config import Config
import os
import skimage.color
import skimage.io

class SixdTestDataset(utils.Dataset):

    def load_sixd_test(self, sixd_image_root, sixd_mask_root):
        self.add_class("sixd", 1, "cup")
        self.add_class("sixd", 2, "carton")
        image_dir_list = sorted(os.listdir(sixd_image_root))
        mask_dir_list = sorted(os.listdir(sixd_mask_root))
        assert len(image_dir_list) == len(mask_dir_list)
        for j in range(len(image_dir_list)):
            image_list = sorted(os.listdir(sixd_image_root + '/' + image_dir_list[j] + '/rgb'))        
            cup_mask_dir = sixd_mask_root + '/' + mask_dir_list[j] + '/mask/obj_01'
            carton_mask_dir = sixd_mask_root + '/' + mask_dir_list[j] + '/mask/obj_02'
            
            cup, carton = False, False
            if os.path.isdir(cup_mask_dir):
                cup_mask_list = sorted(os.listdir(cup_mask_dir))
                cup = True
            if os.path.isdir(carton_mask_dir):
                carton_mask_list = sorted(os.listdir(carton_mask_dir))
                carton = True
            for i in range(len(image_list)):
                # Load scene images
                from scipy import misc
                rgbfile = sixd_image_root + '/' + image_dir_list[j] + '/rgb/' + image_list[i]
                assert os.path.isfile(rgbfile)
                img = misc.imread(rgbfile)
                
                mask, class_ids = np.array, np.array

                if cup and carton:
                    cup_mask = np.load(cup_mask_dir + '/' + cup_mask_list[i])
                    carton_mask = np.load(carton_mask_dir + '/' + cup_mask_list[i])
                    mask = np.concatenate([cup_mask, carton_mask], axis=2)
                    class_ids = np.concatenate([np.ones(cup_mask.shape[2]), np.full(carton_mask.shape[2], 2)], axis=0)
                else:
                    if cup:
                        mask = np.load(cup_mask_dir + '/' + cup_mask_list[i])
                        class_ids = np.ones(mask.shape[2])
                    elif carton:
                        mask = np.load(carton_mask_dir + '/' + carton_mask_list[i])
                        class_ids = np.ones(mask.shape[2]) + 1         
                
                self.add_image("sixd", image_id=i, path=rgbfile, masks=mask, class_ids=class_ids)

            
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def image_reference(self, image_id):
        """Return the image_def data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sixd":
            return info["sixd"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for cups/cartons of the given image ID.
        """
        info = self.image_info[image_id]
        mask, class_ids = info['masks'], info['class_ids']
        
        return mask, class_ids.astype(np.int32)

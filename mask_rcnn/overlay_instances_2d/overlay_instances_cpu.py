import glob
from datetime import datetime
import numpy as np
from PIL import Image
import cv2


def generate_image_def(bg_path, instance_paths, class_ids, border_ratio=0.1,
                       min_scale=0.4, max_scale=1.2):
    with Image.open(bg_path) as img:
        w, h = img.size
    border = border_ratio * min(w, h)
    N = len(instance_paths)
    x = np.random.uniform(border, w - border, N)
    y = np.random.uniform(border, h - border, N)
    scales = np.random.uniform(min_scale, max_scale, N)
    rotations = np.random.uniform(0, 360, N)

    instances = []
    for i, instance_path in enumerate(instance_paths):
        with Image.open(instance_path) as img:
            iw, ih = img.size
        bg_instance_ratio = min(w, h) / min(iw, ih)
        instances.append({
            'path': instance_path, 'size': (iw, ih),
            'class': class_ids[i],
            'x': x[i], 'y': y[i],
            's': scales[i] * bg_instance_ratio,
            'r': rotations[i]
        })

    image_def = {
        'background': {'path': bg_path, 'size': (w, h)},
        'instances': instances
    }

    masks = generate_masks(image_def)

    visible_instances = []
    for i, area in enumerate(masks.sum(axis=(0, 1))):
        if area > 20 ** 2:
            visible_instances.append(instances[i])
    image_def['instances'] = visible_instances

    return image_def


def get_transform_from_inst(inst):
    (w, h), x, y, s, r = (inst[k] for k in ['size', 'x', 'y', 's', 'r'])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), r, s)
    M[0, 2] += x - w / 2
    M[1, 2] += y - h / 2
    return M


def generate_image(img_def):
    dst = cv2.imread(img_def['background']['path']).astype(np.uint16)
    h, w, _ = dst.shape

    instances = img_def['instances']
    for inst in instances:
        src = cv2.imread(inst['path'], cv2.IMREAD_UNCHANGED)
        src = cv2.warpAffine(src, get_transform_from_inst(inst), (w, h))
        dst = (dst * (255 - src[:, :, [3]])) // 255  # alpha from src
        dst += src[:, :, :3]  # works because images are alpha pre multiplied

    return cv2.cvtColor(dst.astype(np.uint8), cv2.COLOR_BGR2RGB)


def generate_masks(img_def):
    w, h = img_def['background']['size']

    instances = img_def['instances']
    masks = np.zeros((h, w, len(instances)), np.bool)
    cum_mask = np.zeros((h, w), np.bool)

    for i, inst in reversed(list(enumerate(instances))):
        mask = cv2.imread(inst['path'], cv2.IMREAD_UNCHANGED)[:, :, 3]
        masks[:, :, i] = cv2.warpAffine(mask, get_transform_from_inst(inst), (w, h))
        masks[:, :, i] &= ~cum_mask
        cum_mask |= masks[:, :, i]

    return masks


def main():
    instancefiles = glob.glob('./instances/**/*.png')

    start = datetime.now()
    img_def = generate_image_def(
        bg_path='./bg.jpg',
        instance_paths=np.random.choice(instancefiles, 10),
        class_ids=np.arange(10),
    )
    print('Def gen took:', datetime.now() - start)

    start = datetime.now()
    img = generate_image(img_def)
    print('Img gen took:', datetime.now() - start)

    start = datetime.now()
    masks = generate_masks(img_def)
    print('Masks gen took:', datetime.now() - start)

    from matplotlib import pyplot as plt

    rows, cols = 3, 3
    plt.subplot(rows, cols, 1)
    plt.imshow(img)
    plt.title('img')
    plt.axis('off')
    plt.subplot(rows, cols, 2)
    zeros = np.zeros((*masks.shape[:2], 1), np.bool)
    depth = np.argmax(np.concatenate((zeros, masks), axis=2), axis=2)
    plt.imshow(depth)
    plt.axis('off')
    for i in range(min((rows - 1) * cols, masks.shape[2])):
        mask_i = masks.shape[2] - i - 1
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(masks[:, :, mask_i])
        plt.title('mask ' + str(mask_i + 1))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

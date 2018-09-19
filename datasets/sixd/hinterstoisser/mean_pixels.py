import numpy as np
from skimage import io
from glob import glob

rgb_image_paths = glob('test/*/rgb/*.png')
depth_image_paths = glob('test/*/depth/*.png')

N = 1000
mean = 0
std = 0

for img_path in np.random.choice(rgb_image_paths, N):
    img = io.imread(img_path)
    img = img.reshape((*img.shape[:2], -1))
    mean += np.mean(img, axis=(0, 1)) / N
    std += np.std(img, axis=(0, 1)) / N

print(mean, std)

import numpy as np
from skimage import io
from glob import glob

depth_image_paths = glob('test/*/depth/*.png')

N = 1000
mean = 0
std = 0

for img_path in np.random.choice(depth_image_paths, N):
    img = io.imread(img_path)
    mean += np.mean(img) / N
    std += np.std(img) / N

print(mean, std)

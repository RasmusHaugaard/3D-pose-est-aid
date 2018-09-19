from random import shuffle
from glob import glob
import sys
from pathlib import Path

dataset_dir = sys.argv[1]
N = int(sys.argv[2])

rgb_image_paths = glob(str(Path(dataset_dir) / 'test') + '/*/rgb/*.png')

shuffle(rgb_image_paths)

for i in range(N):
    print(rgb_image_paths[i])


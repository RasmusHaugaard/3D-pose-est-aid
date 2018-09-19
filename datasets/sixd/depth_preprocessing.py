import numpy as np
from pathlib import Path
from skimage.filters.rank import median
from skimage.morphology import square
from skimage import io
import multiprocessing as mp
import argparse

def reconstruct_missing_values(depth_path):
    out = io.imread(str(depth_path))
    while np.count_nonzero(out == 0) > 0:
        out = np.where((out > 0), out, median(out, selem=square(5), mask=(out > 0)))
    io.imsave(depth_path.parent.parent / 'depth_reconstructed' / depth_path.name, out)
    print(depth_path.name + ' saved')

def main():
    po = argparse.ArgumentParser()
    po.add_argument("root", default="hinterstoisser/test/02/depth", type=str, help="path for depth images")
    args = po.parse_args()
    depth_dir = Path(args.root)
    assert depth_dir.is_dir()
    depth_paths = list(depth_dir.glob('*.png'))
    with mp.Pool(mp.cpu_count()) as p:
        p.map(reconstruct_missing_values, depth_paths)

if __name__ == '__main__':
    main()
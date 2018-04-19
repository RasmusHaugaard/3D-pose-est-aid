import subprocess
from glob import glob
import shutil
from pathlib import Path
import numpy as np
import cv2

from rgbd_overlay import rgbd_overlay

scene_count = 300
img_per_scene = 10
total_img = scene_count * img_per_scene


def main():
    blender_out_dir = Path('./blender_out')

    if blender_out_dir.exists():
        shutil.rmtree(str(blender_out_dir))
    blender_out_dir.mkdir()

    subprocess.run([
        "blender",
        "-b", "gen_obj_cloud.blend",
        "-P", "gen_obj_cloud.py",
        "--",
        "--scene_count", str(scene_count),
        "--img_per_scene", str(img_per_scene),
    ])

    rgba_paths = glob('./blender_out/*/*/rgba*.png')
    mask_paths = [p.replace('rgba', 'mask') for p in rgba_paths]
    depth_paths = [p.replace('rgba', 'depth') for p in rgba_paths]
    bg_rgb_paths = np.random.choice(glob('../hinterstoisser/test/*/rgb/*.png'), total_img)
    bg_depth_paths = [str(p).replace('rgb', 'depth') for p in bg_rgb_paths]

    out_dir = Path('./out')
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir()
    (out_dir / 'rgb').mkdir()
    (out_dir / 'depth').mkdir()
    (out_dir / 'mask').mkdir()

    for i in range(total_img):
        rgb, depth = rgbd_overlay(
            bg_rgb_paths[i], bg_depth_paths[i],
            rgba_paths[i], depth_paths[i],
        )
        cv2.imwrite('./out/rgb/' + str(i) + '.png', rgb)
        cv2.imwrite('./out/depth/' + str(i) + '.png', depth)
        shutil.move(mask_paths[i], './out/mask/' + str(i) + '.png')


if __name__ == '__main__':
    main()

import numpy as np
import cv2


def rgbd_overlay(bg_rgb_path, bg_depth_path, fg_rgba_path, fg_depth_path):
    # load bg and fg
    bg_rgb = cv2.imread(bg_rgb_path, cv2.IMREAD_UNCHANGED)
    bg_depth = cv2.imread(bg_depth_path, cv2.IMREAD_UNCHANGED)
    fg_rgba = cv2.imread(fg_rgba_path, cv2.IMREAD_UNCHANGED)
    fg_depth = cv2.imread(fg_depth_path, cv2.IMREAD_UNCHANGED)

    # rgb channel overlay
    fg_a = fg_rgba[:, :, [3]].astype(np.uint16)
    fg_rgb = fg_rgba[:, :, 0:3]
    rgb = fg_rgb * fg_a + bg_rgb * (255 - fg_a)
    rgb = (rgb // 255).astype(np.uint8)

    # depth channel overlay
    fg_a = fg_depth != 0
    fg_max = fg_depth.max()
    bg_min = bg_depth[(bg_depth * fg_a).nonzero()].min()

    # pull/push the background depth close to the subject
    bg_depth[bg_depth.nonzero()] += fg_max - bg_min
    depth = fg_depth + bg_depth * np.logical_not(fg_a)

    return rgb, depth

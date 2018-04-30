import argparse
import os
import yaml
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy import misc
from scipy.ndimage import morphology
from PIL import Image

import multiprocessing as mp
import functools


def load_model(obj_id, models_dir, models={}):
    path = '{}/obj_{:02}.ply'.format(models_dir, obj_id)
    if not path in models:
        print('Loading object file {}...'.format(path))
        data = PlyData.read(path)
        models[path] = np.stack([
            data['vertex']['x'],
            data['vertex']['y'],
            data['vertex']['z'],
            np.ones(len(data['vertex']['x'])),
        ])
    return models[path]


def generate_model_depth_image(model_vertices, obj_gt, K, img_shape):
    m2c = np.vstack((
        np.hstack((
            np.asarray(obj_gt['cam_R_m2c']).reshape((3, 3)),
            np.asarray(obj_gt['cam_t_m2c']).reshape((3, 1))
        )),
        np.array([0, 0, 0, 1])
    ))
    c2p = np.vstack((
        np.insert(K, 2, (0, 0, 1), axis=1),
        np.array([0, 0, 0, 1])
    ))
    c2p[2, 3] = 0

    verts = m2c @ model_vertices
    verts[0:2] /= verts[2]  # Normalize xy by z
    verts = c2p @ verts

    xy = np.round(verts[0:2]).astype(np.int)
    z = verts[2]

    # Remove pixels beyond image borders
    mask = np.logical_and(
        np.logical_and(xy[0, :] >= 0, xy[0, :] < img_shape[1]),
        np.logical_and(xy[1, :] >= 0, xy[1, :] < img_shape[0])
    )
    xy = xy[:, mask]
    z = z[mask]

    # generate the instance depth image
    inst_z = np.empty(img_shape, dtype=np.float)
    inst_z.fill(10 ** 6)
    for j in range(z.shape[0]):
        y, x = xy[1, j], xy[0, j]
        if z[j] < inst_z[y, x]:
            inst_z[y, x] = z[j]

    return inst_z


def handle_scene(scene_dir, models_dir, obj_group_size):
    rgb_dir = scene_dir / 'rgb'
    assert rgb_dir.is_dir()
    print('Loading scene file list from directory {}...'.format(rgb_dir))
    rgb_list = sorted(list(rgb_dir.glob('*.png')))
    print('\tGot {} files from {} to {}'.format(len(rgb_list), rgb_list[0], rgb_list[-1]))

    print('Loading GT poses from {}...'.format(scene_dir / 'gt.yml'))
    with (scene_dir / 'gt.yml').open('r') as f:
        scenes_objs_gt = [objs_gt for _, objs_gt in yaml.load(f).items()]

    infofile = scene_dir / 'info.yml'
    print('Loading camera info from {}...'.format(infofile))
    with infofile.open('r') as f:
        cam_Ks = [np.array(cam_info['cam_K']).reshape(3, 3)
                  for _, cam_info in yaml.load(f).items()]

    assert len(scenes_objs_gt) == len(cam_Ks)

    outdir = scene_dir / 'mask'
    if not outdir.is_dir():
        outdir.mkdir()

    seqid_string = scene_dir.name
    print('Traversing scene in sequence {}...'.format(seqid_string))

    for rgb_file, cam_K, objs_gt in zip(rgb_list, cam_Ks, scenes_objs_gt):
        with Image.open(rgb_file) as img:
            img_shape = img.size[1], img.size[0]

        img_z = np.empty(img_shape, dtype=np.float)
        img_z.fill(10 ** 6)
        img_mask = np.zeros(img_shape, dtype=np.uint8)
        run_inst_ids = {}
        for obj_gt in objs_gt:
            obj_id = obj_gt['obj_id']
            model = load_model(obj_id, models_dir)

            inst_z = generate_model_depth_image(model, obj_gt, cam_K, img_shape)
            # remove salt noise from non-dense model / view ratio
            inst_z = morphology.grey_opening(inst_z, size=(3, 3))

            inst_mask = inst_z < img_z
            img_z[inst_mask] = inst_z[inst_mask]
            local_inst_id = run_inst_ids.get(obj_id, 0)
            assert local_inst_id < obj_group_size
            global_inst_id = obj_id * obj_group_size + local_inst_id
            assert global_inst_id < 256
            img_mask[inst_mask] = global_inst_id
            run_inst_ids[obj_id] = local_inst_id + 1

        # Save result
        outfile = outdir / rgb_file.name
        print('\tSaving output file {} with {} annotated instances...'.format(outfile, len(objs_gt)))
        misc.imsave(str(outfile), img_mask)


def main():
    po = argparse.ArgumentParser()
    po.add_argument("root", default="/path/to/sixd/dataset", type=str, help="root path of your dataset")
    po.add_argument("--models-dir", '-o', default="models", type=str, help="relative path to directory with models")
    po.add_argument("--scene-glob", '-s', default="test/*/", type=str, help="relative path to directory with scenes")
    po.add_argument("--obj-group-size", default=1, type=int, help="Max instances per object per image")
    args = po.parse_args()

    root = Path(args.root)
    assert root.is_dir()

    scene_dirs = sorted(list(filter(lambda f: f.is_dir(), root.glob(args.scene_glob))))

    models_dir = root / args.models_dir
    assert models_dir.is_dir()

    def _handle_scene(scene_dir):
        handle_scene(scene_dir, models_dir, args.obj_group_size)

    with mp.Pool(mp.cpu_count()) as p:
        p.map(
            functools.partial(handle_scene, models_dir=models_dir, obj_group_size=args.obj_group_size),
            scene_dirs
        )


if __name__ == '__main__':
    main()

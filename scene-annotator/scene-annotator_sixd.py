#!/usr/bin/env python

import numpy, os, yaml

import sys

sys.path.append('../covis/build/lib')

from covis import *


root = "/home/mjakobsen/workspace/scene-annotator"
object = "models/obj_01.ply"
scene_dir = "../../../rhaugaard/shared/3D-pose-est-aid/datasets/sixd/doumanoglou/test/02"
threshold =10
output_dir = "./output"

model = util.load(root + '/' + object)
idot = object.rfind('.')
objid_string = object[idot - 2:idot]
objid = int(objid_string)
scenedir = root + '/' + scene_dir
assert os.path.isdir(scenedir)
scenedirrgb = scenedir + '/rgb'
assert os.path.isdir(scenedirrgb)
rgblist = sorted(os.listdir(scenedirrgb))
scenedird = scenedir + '/depth'
assert os.path.isdir(scenedird)
dlist = sorted(os.listdir(scenedird))
assert len(rgblist) == len(dlist)
gtfile = scenedir + '/gt.yml'
assert os.path.isfile(gtfile)
with open(gtfile, 'r') as f:
    gtdata = yaml.load(f)
infofile = scenedir + '/info.yml'
assert os.path.isfile(infofile)
with open(infofile, 'r') as f:
    camdata = yaml.load(f)
outdir = output_dir + '/' + root[root.rfind('/') + 1:] + '/' + scene_dir[-2:] + '/mask/obj_' + objid_string
if not os.path.isdir(outdir):
    os.makedirs(outdir)
assert len(rgblist) == len(gtdata) == len(camdata)
seqid_string = scene_dir[-2:]

for i in range(len(rgblist)):
    # Load scene images
    from scipy import misc
    rgbfile = scenedirrgb + '/' + rgblist[i]
    assert os.path.isfile(rgbfile)
    dfile = scenedird + '/' + dlist[i]
    assert os.path.isfile(dfile)
    img = misc.imread(rgbfile)
    depth = misc.imread(dfile, mode='I')

    # Load intrinsic matrix
    assert camdata[i]['depth_scale'] == 1
    K = camdata[i]['cam_K']
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]

    # Load GT poses for object
    Tlist = gtdata[i]
    tmp = []
    for T in Tlist:
        if T['obj_id'] == objid:
            tmp.append(T)
    Tlist = tmp

    # Generate masks for each object instance, put them into this binary image (0 for bg, 255 for fg)

    img_masked_concat = numpy.zeros((img.shape[0], img.shape[1], len(Tlist)), dtype=bool)
    k = 0
    for T in Tlist:
        img_masked = numpy.zeros((img.shape[0], img.shape[1]), dtype=bool)
        # Format pose
        R = numpy.asarray(T['cam_R_m2c']).reshape((3, 3))
        t = numpy.asarray(T['cam_t_m2c']).reshape((3, 1))
        Ti = numpy.vstack((numpy.hstack((R, t)), numpy.array([0, 0, 0, 1])))
        # Transform the object coordinates into the scene
        xyz = model.cloud.array()[0:3, :]
        xyz = numpy.matmul(R, xyz) + numpy.tile(t, xyz.shape[1])
        # Project to pixel coordinates
        xy = xyz[0:2, :] / xyz[2, :]  # Normalize by z
        xy[0, :] = fx * xy[0, :] + cx
        xy[1, :] = fy * xy[1, :] + cy
        xy = numpy.round(xy).astype(int)
        z = xyz[2, :]  # Maintain depth for use below

        # Remove pixels beyond image borders
        mask = numpy.logical_and(numpy.logical_and(xy[0, :] >= 0, xy[0, :] <= img.shape[1] - 1),
                                 numpy.logical_and(xy[1, :] >= 0, xy[1, :] <= img.shape[0] - 1))
        xy = xy[:, mask]
        z = z[mask]

        # Remove pixels behind scene data
        if threshold > 0:
            mask = numpy.zeros_like(z, dtype=bool)
            for j in range(z.shape[0]):
                if depth[xy[1, j], xy[0, j]] > 0:
                    if abs(depth[xy[1, j], xy[0, j]] - z[j]) < threshold:
                        #                    if depth[xy[1,j], xy[0,j]] < z[j] - threshold:
                        mask[j] = True
            xy = xy[:, mask]
            z = z[mask]

        img_masked[xy[1, :], xy[0, :]] = 1  # Row index (y) comes first

    # Remove pepper noise
        from skimage import morphology
        selem = numpy.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
        img_masked = morphology.closing(img_masked)
        img_masked = morphology.dilation(img_masked, selem)
        img_masked = morphology.dilation(img_masked, selem)
        img_masked = morphology.remove_small_holes(img_masked, 10000)
        img_masked = morphology.erosion(img_masked, selem)
        img_masked = morphology.erosion(img_masked, selem)
        img_masked_concat[:, :, k] = img_masked

        k = k + 1
    outfile = outdir + '/' + rgblist[i]
    numpy.save(outfile + '.npy', img_masked_concat)
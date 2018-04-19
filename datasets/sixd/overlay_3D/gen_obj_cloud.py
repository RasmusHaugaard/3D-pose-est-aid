# noinspection PyUnresolvedReferences
import bpy
# noinspection PyUnresolvedReferences
import mathutils

from random import uniform, randint, choice, shuffle
import math
import sys
import argparse


X, Y, Z = 0, 1, 2
objs = bpy.data.objects
scn = bpy.data.scenes['RenderScene']


def rand_euler():
    z = uniform(0, 2 * math.pi)
    # adjust for equal sphere area probability
    x = math.acos(1 - uniform(0, 2))
    return mathutils.Euler((x, 0, z), 'XYZ')


def set_rand_lights():
    light_parent_objs = objs['LightParents'].children
    for p in light_parent_objs:
        p.rotation_euler = rand_euler()

    light_objs = [p.children[0] for p in light_parent_objs]
    for l in light_objs:
        l.location[Z] = 7

    for i in range(len(light_objs)):
        lamp = bpy.data.lamps[str(i)]
        lamp.size = uniform(1, 7)    


def set_rand_camera():
    cam_loc = objs['Camera'].location
    cam_loc[X] = uniform(-2.5, 2.5)
    cam_loc[Y] = uniform(-2, 2)
    cam_loc[Z] = uniform(10, 20)
    objs['CameraParent'].rotation_euler = rand_euler()


def set_rand_scene():
    instances_parent = objs['Instances']
        
    for obj in instances_parent.children:
        objs.remove(obj, True)
    
    min_objs, max_objs = 20, 40
    total_objs_count = randint(min_objs, max_objs)
    positives_count = randint(total_objs_count // 3, total_objs_count)
    negatives_count = total_objs_count - positives_count
    cup_count = randint(0, positives_count)
    carton_count = positives_count - cup_count
    
    cup, carton = objs['Cup'], objs['Carton']
    negatives_stock = objs['NegativesStock'].children
    
    instances = []
    for _ in range(negatives_count):
        instances.append(choice(negatives_stock).copy())
    for i in range(cup_count):
        obj = cup.copy()
        obj.pass_index = 40 + i
        instances.append(obj)
    for i in range(carton_count):
        obj = carton.copy()
        obj.pass_index = 80 + i
        instances.append(obj)
    shuffle(instances)
    
    scn.frame_set(0)
    box_size = math.ceil(len(instances) ** (1./3.))
    for i, obj in enumerate(instances):
        z = i // (box_size * box_size)
        i -= z * box_size * box_size
        y = i // box_size
        x = i - y * box_size
        
        for i, l in ((0, x), (1, y), (2, z)):
            obj.location[i] = (l - (box_size-1) / 2) * 3
            
        obj.rotation_euler = rand_euler()
        scn.objects.link(obj)
        obj.parent = instances_parent
        scn.rigidbody_world.group.objects.link(obj)

    for i in range(1, 250):
        scn.frame_set(i)
    

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_count', action="store", dest="scene_count", type=int, required=True)
    parser.add_argument('--img_per_scene', action="store", dest="img_per_scene", type=int, required=True)
    args = parser.parse_args(argv)
    scene_count, img_per_scene = args.scene_count, args.img_per_scene

    total_img = scene_count * img_per_scene
    for scene_i in range(scene_count):
        print('Setting random scene', scene_i+1, 'of', scene_count, '..')
        set_rand_scene()
        for img_i in range(img_per_scene):
            set_rand_lights()
            set_rand_camera()
            print('Rendering', scene_i * img_per_scene + img_i + 1, 'of', total_img, '..')
            fo = scn.node_tree.nodes["File Output"]
            fo.base_path = "//blender_out/" + str(scene_i) + "/" + str(img_i)
            bpy.ops.render.render()


if __name__ == '__main__':
    main()

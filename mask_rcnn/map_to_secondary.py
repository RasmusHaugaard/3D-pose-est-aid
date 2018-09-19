import h5py
import re
import numpy as np

f = h5py.File('mask_rcnn_hinterstoisser_0100_secondary.h5', mode='r+')
if 'layer_names' not in f.attrs and 'model_weights' in f:
    f = f['model_weights']

layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

if True:
    r = re.compile('(conv1)|(res.*)|(bn.*)')
    for i, name in enumerate(layer_names):
        if r.fullmatch(name):
            new_name = 'secondary_' + name
            f[new_name] = f.pop(name)
            layer_names[i] = new_name

    f.attrs['layer_names'] = np.array(layer_names, dtype='S')

for key in f.keys():
    print(key)

for name in f.attrs['layer_names']:
    print('LAYER:', name)
    for key in f[name].attrs['weight_names']:
        print('W:', key)


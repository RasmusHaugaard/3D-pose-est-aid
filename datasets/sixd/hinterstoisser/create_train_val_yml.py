import yaml
from pathlib import Path
import random

validation_size = 100

with open('./test_set_v1.yml', 'r') as f:
    data = yaml.load(f)

test_dir = Path('./test')

train = {}
val = {}

for scene_index, test_image_indices in data.items():
    scene_dir = test_dir / '{:02}'.format(scene_index)
    scene_images = list(map(str, (scene_dir / 'rgb').glob('*.png')))

    all_set = set([int(file_name[-8:-4]) for file_name in scene_images])
    test_set = set(test_image_indices)
    train_val_indices = list(all_set.difference(test_set))
    random.shuffle(train_val_indices)
    val_indices = train_val_indices[:validation_size]
    train_indices = train_val_indices[validation_size:]
    val[scene_index] = val_indices
    train[scene_index] = train_indices

    val_set = set(val_indices)
    train_set = set(train_indices)

    assert len(all_set) == len(test_set) + len(val_set) + len(train_set)
    assert test_set.isdisjoint(train_set | val_set)
    assert val_set.isdisjoint(train_set | test_set)
    assert train_set.isdisjoint(val_set | test_set)


for file, data in (("./val_set.yml", val), ("./train_set.yml", train)):
    with open(file, 'w') as f:
        yaml.dump(data, f)

import os
import random
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from tqdm import tqdm
import math
import json

# Configuration
coco_path = './MsCoCo'
ann_file = os.path.join(coco_path, 'annotations/instances_train2017.json')
image_dir = os.path.join(coco_path, 'train2017')
output_dir = './SceneObject'
places_root = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/datasets/places365_standard/data_large'

os.makedirs(output_dir, exist_ok=True)

selected_classes = ['boat', 'airplane', 'truck', 'dog', 'zebra', 'horse', 'bird', 'train', 'bus', 'motorcycle']  # example
scene_classes = ['beach', 'canyon', 'building_facade', 'staircase', 'desert_sand', 'crevasse', 'bamboo_forest', 'shower', 'ball_pit', 'kasbah']
object_to_scene = dict(zip(selected_classes, scene_classes))

# Bias coefficients for 3 environments
env_biases = [0.9, 0.7, 0.0]  # env1, env2, test

# Load COCO
coco = COCO(ann_file)

def get_images_by_class(cat_name, num_samples):
    cat_id = coco.getCatIds(catNms=[cat_name])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    return random.sample(img_ids, num_samples)

def sample_scene_image(scene_label, places_root):
    # e.g., places_root/kitchen/xyz.jpg
    scene_dir = os.path.join(places_root, f'{scene_label[0]}', scene_label)
    candidates = os.listdir(scene_dir)
    img_path = os.path.join(scene_dir, random.choice(candidates))
    return Image.open(img_path).convert("RGB")

def composite_object_on_scene(obj_img, ann, bg_img, size=(224, 224)):
    mask = coco.annToMask(ann)
    obj_np = np.array(obj_img)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    obj_rgb = obj_np * mask_3d

    bg_np = np.array(bg_img.resize(obj_img.size))
    composite = np.where(mask_3d == 1, obj_rgb, bg_np)
    return Image.fromarray(composite.astype(np.uint8)).resize(size)

metadata = {'env_0': [], 'env_1': [], 'env_2': []}
counter = {'env_0': 0, 'env_1': 0, 'env_2': 0}

for class_name in selected_classes:
    print(f'Building class {class_name}')
    img_ids = get_images_by_class(class_name, 1500)  # adjust number if needed
    cat_id = coco.getCatIds(catNms=[class_name])[0]
    for idx, img_id in tqdm(enumerate(img_ids)):
        env_id = math.floor(idx / 600)
        bias = env_biases[env_id]
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        img = Image.open(os.path.join(image_dir, img_info['file_name'])).convert("RGB")

        for ann in anns:
            # Skip annotations without segmentation
            if 'segmentation' not in ann or ann['area'] < 1000:
                continue

            # Choose background color based on bias
            if random.random() < bias:
                scene_class = object_to_scene[class_name]
            else:
                scene_class = random.choice(scene_classes)
            scene_img = sample_scene_image(scene_class, places_root)

            composed_img = composite_object_on_scene(img, ann, scene_img)
            label = selected_classes.index(class_name)

            env_name = f'env_{env_id}'
            idx_count = counter[env_name]
            image_filename = f'{class_name}_{idx_count:04d}.png'
            image_path = os.path.join(output_dir, env_name, 'images', image_filename)
            if not os.path.exists(image_path):
                os.makedirs(os.path.join(output_dir, env_name, 'images'), exist_ok=True)
            composed_img.save(image_path)

            metadata[env_name].append({
                'file_name': os.path.join(env_name, 'images', image_filename),
                'label': label,
                'bg_scene': [scene_class]
            })
            counter[env_name] += 1
            break  # use only one object per image for simplicity

# Save metadata as JSON files
for env_name in metadata:
    json_path = os.path.join(output_dir, env_name, 'metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata[env_name], f, indent=2)


# Now you can save or use `dataset` [(PIL.Image, label, env_id), ...]

import os
import random
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from tqdm import tqdm
import math
import pickle
import json

# Configuration
coco_path = './MsCoCo'
ann_file = os.path.join(coco_path, 'annotations/instances_train2017.json')
image_dir = os.path.join(coco_path, 'train2017')
output_dir = './ColoredObject'

os.makedirs(output_dir, exist_ok=True)

selected_classes = ['boat', 'airplane', 'truck', 'dog', 'zebra', 'horse', 'bird', 'train', 'bus', 'motorcycle']  # example
background_colors = [(0, 100, 0), (188, 143, 143), (255, 0, 0), (255, 215, 0), (0, 255, 0), (65, 105, 225), (0, 225, 225), (0, 0, 255), (255, 20, 147), (160, 160, 160)]

# One-to-one mapping
class_to_color = dict(zip(selected_classes, background_colors))

# Bias coefficients for 3 environments
env_biases = [0.8, 0.6, 0.7, 0.0]  # env1, env2, test

# Load COCO
coco = COCO(ann_file)

def get_images_by_class(cat_name, num_samples):
    cat_id = coco.getCatIds(catNms=[cat_name])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    return random.sample(img_ids, num_samples)


def composite_object_with_mask(img, ann, bg_color=(255, 255, 255), size=(224, 224)):
    mask = coco.annToMask(ann)  # binary mask, shape (H, W)
    img_np = np.array(img)

    # Mask the image
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    obj_rgb = img_np * mask_3d
    obj_img = Image.fromarray(obj_rgb.astype(np.uint8))

    # Create colored background
    bg = Image.new("RGB", img.size, bg_color)
    bg_np = np.array(bg)

    # Composite
    final = np.where(mask_3d == 1, obj_rgb, bg_np)
    result_img = Image.fromarray(final.astype(np.uint8)).resize(size)
    return result_img

for i in range(4):
    env_dir = os.path.join(output_dir, f'env_{i}')
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(os.path.join(env_dir, 'images'), exist_ok=True)

metadata = {'env_0': [], 'env_1': [], 'env_2': [], 'env_3': []}
counter = {'env_0': 0, 'env_1': 0, 'env_2': 0, 'env_3': 0}

for class_name in selected_classes:
    print(f'Building class {class_name}')
    img_ids = get_images_by_class(class_name, 1500)
    cat_id = coco.getCatIds(catNms=[class_name])[0]
    ctft_class_choices = [x for x in selected_classes if x != class_name]

    for idx, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
        if idx < 480:
            env_id = 0
        elif idx < 960:
            env_id = 1
        elif idx < 1200:
            env_id = 2
        else:
            env_id = 3
        bias = env_biases[env_id]
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns: continue

        img = Image.open(os.path.join(image_dir, img_info['file_name'])).convert("RGB")
        for ann in anns:
            if 'segmentation' not in ann or ann['area'] < 1000:
                continue

            # Set background color (biased or random)
            if random.random() < bias:
                bg_color = class_to_color[class_name]
            else:
                bg_color = random.choice(background_colors)

            label = selected_classes.index(class_name)
            env_name = f'env_{env_id}'
            idx_count = counter[env_name]
            base_filename = f'{class_name}_{idx_count:04d}'
            base_path = os.path.join(output_dir, env_name, 'images')

            # ---- Original ----
            composed_img = composite_object_with_mask(img, ann, bg_color=bg_color)
            orig_filename = f'{base_filename}.png'
            composed_img.save(os.path.join(base_path, orig_filename))

            # ---- CTFT Background (same foreground, new background) ----
            ctft_bg_color = random.choice([x for x in background_colors if x != bg_color])
            ctft_bg_img = composite_object_with_mask(img, ann, bg_color=ctft_bg_color)
            ctft_bg_filename = f'{base_filename}_bg.png'
            ctft_bg_img.save(os.path.join(base_path, ctft_bg_filename))

            # ---- CTFT Foreground (new object of different class) ----
            ctft_class = random.choice(ctft_class_choices)
            ctft_cat_id = coco.getCatIds(catNms=[ctft_class])[0]
            ctft_img_id = random.choice(get_images_by_class(ctft_class, 1))
            ctft_img_info = coco.loadImgs(ctft_img_id)[0]
            ctft_ann_ids = coco.getAnnIds(imgIds=[ctft_img_id], catIds=[ctft_cat_id], iscrowd=False)
            ctft_anns = coco.loadAnns(ctft_ann_ids)
            if not ctft_anns: continue
            ctft_img = Image.open(os.path.join(image_dir, ctft_img_info['file_name'])).convert("RGB")
            for ctft_ann in ctft_anns:
                if 'segmentation' not in ctft_ann or ctft_ann['area'] < 1000:
                    continue
                ctft_fg_img = composite_object_with_mask(ctft_img, ctft_ann, bg_color=bg_color)
                ctft_fg_filename = f'{base_filename}_fg.png'
                ctft_fg_img.save(os.path.join(base_path, ctft_fg_filename))
                break

            # ---- Update Metadata ----
            metadata[env_name].append({
                'file_name': os.path.join(env_name, 'images', orig_filename),
                'cat_id': cat_id,
                'object_id': img_id,
                'label': label,
                'bg_color': bg_color,
                'ctft_fg': os.path.join(env_name, 'images', ctft_fg_filename),
                'ctft_fg_cat_id': ctft_cat_id,
                'ctft_fg_img_id': ctft_img_id,
                'ctft_bg': os.path.join(env_name, 'images', ctft_bg_filename),
                'ctft_bg_color': ctft_bg_color
            })
            counter[env_name] += 1
            break

for env_name in metadata:
    json_path = os.path.join(output_dir, env_name, 'metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata[env_name], f, indent=2)
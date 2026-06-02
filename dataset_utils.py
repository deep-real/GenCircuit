import copy
import json
import pickle
import os
import random
from collections import defaultdict

import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import timm
import torch
from imagecorruptions import corrupt
import wilds
from torchvision.transforms import transforms
from transformers import AutoImageProcessor

import pandas as pd

from dataset import VisionEAPDataset

import numpy as np
import torchvision
from PIL import Image, ImageFilter, ImageOps

class CIFAR10_C(torchvision.datasets.CIFAR10):

    def __init__(self, root, data_type=None, severity=1, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(root + "/" + data_type + '.npy')
        labels = np.load(root + "/" + 'labels.npy')

        self.data = data[(severity - 1) * 10000: (severity) * 10000]
        self.targets = labels[(severity - 1) * 10000: (severity) * 10000].astype(np.int_)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SubsetFolder:
    def __init__(self, dataset, class_id):
        if isinstance(class_id, int):
            samples = np.array(dataset.samples, )[np.array(dataset.targets) == class_id]
        elif isinstance(class_id, list):
            class_id = np.array(class_id)
            mask = np.isin(np.array(dataset.targets), class_id)
            samples = np.array(dataset.samples, )[mask]
        self.samples = [(row[0], int(row[1])) for row in samples.tolist()]

    def __len__(self):
        return len(self.samples)

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, processor=None, transform=None, select_class=None, ctft_class_ranges=None):
        dataset = datasets.ImageFolder(root=root_dir)
        self.processor = processor
        self.transform = transform
        self.ctft_class_ranges = ctft_class_ranges
        if select_class is not None:
            subset_dataset = SubsetFolder(dataset, select_class)
            self.dataset = subset_dataset
            ctft_subset_dataset = SubsetFolder(dataset, ctft_class_ranges)
            self.ctft_dataset = ctft_subset_dataset
            self.ctft_class_index = self._build_class_index(self.ctft_dataset) if ctft_subset_dataset else None
        else:
            self.dataset = dataset

    def _build_class_index(self, dataset):
        class_index = {}
        if hasattr(dataset, 'target_name'):
            for idx, label in enumerate(getattr(dataset, dataset.target_name)):  # or dataset.labels
                label = int(label)
                if label not in class_index:
                    class_index[label] = []
                class_index[label].append(idx)
        else:
            for idx, data in enumerate(dataset.samples):
                if len(data) == 2:
                    image, label = data
                    _ = None
                elif len(data) == 3:
                    image, label, _ = data
                else:
                    raise ValueError("Unexpected number of items returned by dataset")
                label = int(label)  # Ensure consistent key type
                if label not in class_index:
                    class_index[label] = []
                class_index[label].append(idx)
        return class_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")  # Use processor
            return inputs["pixel_values"].squeeze(0), label
        if self.transform is not None:
            inputs = self.transform(image)
            return inputs, label


# CIFAR-10.1 dataset, by Rebecca Roelofs and Ludwig Schmidt
# Copying the utils from there for convenience.

import os
import pathlib
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset, random_split
import torchvision.datasets as datasets
import numpy as np

class CorruptionTransform:
    def __init__(self, corruption='gaussian_noise', severity=3):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, x):
        """
        x: PIL.Image or Tensor (C,H,W) in [0,1]
        returns: PIL.Image (uint8, HxW[C]) after corruption
        """
        if isinstance(x, Image.Image):
            arr = np.array(x)
        elif hasattr(x, "numpy"):  # torch.Tensor
            arr = x.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
        elif isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr.clip(0, 1) * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported input type {type(x)}")

        # handle grayscale
        if arr.ndim == 2:
            arr = arr[..., None]

        # enforce >= 32 px (ImageNet-C requirement)
        if arr.shape[0] < 32 or arr.shape[1] < 32:
            pad_h = max(0, 32 - arr.shape[0])
            pad_w = max(0, 32 - arr.shape[1])
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

        arr_cor = corrupt(arr, corruption_name=self.corruption, severity=self.severity)
        return Image.fromarray(arr_cor)

class MetaShiftDataset(Dataset):
    def __init__(self, split, root_dir, transform=None):
        """
        Args:
            split (str): One of ['train', 'majority-val', 'minority-val']
            root_dir (str): Path to 'data' directory (e.g., '../../experiments/metashift/data')
            transform (callable, optional): Transform to apply to each image
        """
        self.split = split
        self.root_dir = root_dir
        self.transform = transform

        # Load metadata
        metadata_path = os.path.join(root_dir, "metadata.csv")
        metadata_df = pd.read_csv(metadata_path)
        if split != 'train':
            self.samples = metadata_df[metadata_df["split"] != 'train'].reset_index(drop=True)
        else:
            self.samples = metadata_df[metadata_df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row['class'])  # 0 = cat, 1 = dog
        env = int(row['env'])      # 0 = indoor, 1 = outdoor

        return img, label, env

class PACSDataset(Dataset):
    def __init__(self, split_json, split_idx, transform=None):
        self.transform = transform

        # Load split
        with open(split_json, "r") as f:
            all_splits = json.load(f)
        if split_idx == "all":
            self.samples = []
            for key in all_splits:
                self.samples.extend(all_splits[key])
        else:
            self.samples = all_splits[str(split_idx)]  # 0: train, 2: val, 1: test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        # Full path to image file
        img_path = entry["filename"]
        image = Image.open(img_path).convert("RGB")
        label = entry["class"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transform(model_name, train=False):
    model = timm.create_model(model_name, pretrained=True)
    config = timm.data.resolve_model_data_config(model)
    transform_test = timm.data.create_transform(**config, is_training=train)
    return transform_test

# Map from ImageNet renditions indices to ImageNet indices.
r_indices = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105,
             107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187,
             195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267,
             269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327,
             330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393,
             397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476,
             483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658,
             701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883,
             889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
             967, 980, 981, 983, 988]

# Map from ImageNet-A indices to ImageNet indices.
a_indices = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105,
             107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295,
             298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336,
             347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438,
             445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552,
             557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668,
             677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786,
             792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880,
             888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972,
             980, 981, 984, 986, 987, 988]

v2_indices = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117,
              118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136,
              137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 155,
              156, 157, 158, 159, 16, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173, 174,
              175, 176, 177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 19, 190, 191, 192, 193,
              194, 195, 196, 197, 198, 199, 2, 20, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 21, 210, 211, 212,
              213, 214, 215, 216, 217, 218, 219, 22, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 23, 230, 231,
              232, 233, 234, 235, 236, 237, 238, 239, 24, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 25, 250,
              251, 252, 253, 254, 255, 256, 257, 258, 259, 26, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 27,
              270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 28, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
              29, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 3, 30, 300, 301, 302, 303, 304, 305, 306, 307, 308,
              309, 31, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 32, 320, 321, 322, 323, 324, 325, 326, 327,
              328, 329, 33, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 34, 340, 341, 342, 343, 344, 345, 346,
              347, 348, 349, 35, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 36, 360, 361, 362, 363, 364, 365,
              366, 367, 368, 369, 37, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 38, 380, 381, 382, 383, 384,
              385, 386, 387, 388, 389, 39, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 4, 40, 400, 401, 402, 403,
              404, 405, 406, 407, 408, 409, 41, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 42, 420, 421, 422,
              423, 424, 425, 426, 427, 428, 429, 43, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 44, 440, 441,
              442, 443, 444, 445, 446, 447, 448, 449, 45, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 46, 460,
              461, 462, 463, 464, 465, 466, 467, 468, 469, 47, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 48,
              480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 49, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 5,
              50, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 51, 510, 511, 512, 513, 514, 515, 516, 517, 518,
              519, 52, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 53, 530, 531, 532, 533, 534, 535, 536, 537,
              538, 539, 54, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 55, 550, 551, 552, 553, 554, 555, 556,
              557, 558, 559, 56, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 57, 570, 571, 572, 573, 574, 575,
              576, 577, 578, 579, 58, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 59, 590, 591, 592, 593, 594,
              595, 596, 597, 598, 599, 6, 60, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 61, 610, 611, 612, 613,
              614, 615, 616, 617, 618, 619, 62, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 63, 630, 631, 632,
              633, 634, 635, 636, 637, 638, 639, 64, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 65, 650, 651,
              652, 653, 654, 655, 656, 657, 658, 659, 66, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 67, 670,
              671, 672, 673, 674, 675, 676, 677, 678, 679, 68, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 69,
              690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 7, 70, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709,
              71, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 72, 720, 721, 722, 723, 724, 725, 726, 727, 728,
              729, 73, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 74, 740, 741, 742, 743, 744, 745, 746, 747,
              748, 749, 75, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 76, 760, 761, 762, 763, 764, 765, 766,
              767, 768, 769, 77, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 78, 780, 781, 782, 783, 784, 785,
              786, 787, 788, 789, 79, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 8, 80, 800, 801, 802, 803, 804,
              805, 806, 807, 808, 809, 81, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 82, 820, 821, 822, 823,
              824, 825, 826, 827, 828, 829, 83, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 84, 840, 841, 842,
              843, 844, 845, 846, 847, 848, 849, 85, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 86, 860, 861,
              862, 863, 864, 865, 866, 867, 868, 869, 87, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 88, 880,
              881, 882, 883, 884, 885, 886, 887, 888, 889, 89, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 9, 90,
              900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 91, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919,
              92, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 93, 930, 931, 932, 933, 934, 935, 936, 937, 938,
              939, 94, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 95, 950, 951, 952, 953, 954, 955, 956, 957,
              958, 959, 96, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 97, 970, 971, 972, 973, 974, 975, 976,
              977, 978, 979, 98, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 99, 990, 991, 992, 993, 994, 995,
              996, 997, 998, 999]

model_dict = {
    'ViT-B_16-in21k':'vit_base_patch16_224_in21k',
    'ViT-S_16':'vit_small_patch16_224_in21k',
    'ViT-Ti_16':'vit_tiny_patch16_224_in21k',
    'ViT-B_16-clip-openai': 'vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-laion2b': 'vit_base_patch16_clip_224.laion2b',
    'ViT-B_16-mae': 'vit_base_patch16_224.mae',
    'ViT-B_16-dinov2': 'vit_base_patch14_dinov2.lvd142m',
    'ViT-B_16-in1k': 'vit_base_patch16_224.orig_in21k_ft_in1k',
    'ViT-B_16-in21k-hook':'vit_base_patch16_224_in21k',
    'ViT-B_16-scratch': 'vit_base_patch16_224'
}

class CountourStylizeTransform:
    def __init__(self, p=0.5):
        self.p = p
        self.filters = ImageFilter.CONTOUR

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(self.filters)

class EdgeEnhanceStylizeTransform:
    def __init__(self, p=0.5):
        self.p = p
        self.filters = ImageFilter.EDGE_ENHANCE

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(self.filters)

class EmbossStylizeTransform:
    def __init__(self, p=0.5):
        self.p = p
        self.filters = ImageFilter.EMBOSS

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(self.filters)

class CartoonStylizeTransform:
    def __call__(self, img):
        arr = np.array(img)
        # Apply bilateral filter for smooth color regions
        color = cv2.bilateralFilter(arr, d=9, sigmaColor=250, sigmaSpace=250)
        # Detect edges
        edges = cv2.Canny(arr, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        cartoon = cv2.bitwise_and(color, 255 - edges)
        return Image.fromarray(cartoon)


class PaletteStylizeTransform:
    def __call__(self, img):
        arr = np.array(img)
        arr = np.take(np.random.permutation(256), arr)  # remap colors
        return Image.fromarray(arr.astype(np.uint8))

class PosterizeStylizeTransform:
    def __init__(self, bits=2):
        self.bits = bits
    def __call__(self, img):
        return ImageOps.posterize(img, self.bits)

class SolarizeStylizeTransform:
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

class EdgeStylizeTransform:
    def __call__(self, img):
        return ImageOps.invert(img.convert("L").filter(ImageFilter.FIND_EDGES)).convert("RGB")

class ImageNet(Dataset):

    def __init__(self, root, split='train', num_examples=None, transform=None, processor=None, seed=0):
        super().__init__()
        self.data = datasets.ImageFolder(root=root + '/' + split, transform=None)
        self._split = split
        self._num_examples = num_examples
        self._transform = transform
        self._processor = processor
        if self._split in ['imagenet-r', 'renditions']:
            self.valid_indices = r_indices
        elif self._split == 'imagenet-a':
            self.valid_indices = a_indices
        if self._num_examples is not None:
            if self._num_examples > len(self.data):
                raise ValueError('num_examples can be at most the dataset size {len(self.data)}')
            rng = np.random.RandomState(seed=seed)
            self._data_indices = rng.permutation(len(self.data))[:num_examples]

    def __getitem__(self, i):
        if self._num_examples is not None:
            i = self._data_indices[i]
        x, y = self.data[i]
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        if self._processor is not None:
            x = self._processor(images=x, return_tensors="pt")["pixel_values"].squeeze(0)
        if self._split == 'renditions' or self._split == 'imagenet-r':
            y = r_indices[y]
        elif self._split == 'imagenet-a':
            y = a_indices[y]
        elif self._split == 'v2' or self._split == 'imagenetv2-matched-frequency-format-val':
            y = v2_indices[y]
        return x, y

    def __len__(self) -> int:
        return len(self.data) if self._num_examples is None else self._num_examples

def bernoulli_kl(logit1, logit2, eps=1e-7, reduction="batchmean"):
    p = torch.sigmoid(logit1)
    q = torch.sigmoid(logit2)

    p = p.clamp(min=eps, max=1 - eps)
    q = q.clamp(min=eps, max=1 - eps)

    # KL divergence between Bernoulli(p) and Bernoulli(q)
    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    if reduction == "batchmean":
        return kl.mean()

def get_failure_list(failure_path, dataset, model):
    if os.path.exists(failure_path):
        with open(failure_path, 'rb') as f:
            failure_indices = pickle.load(f)
    else:
        failure_indices = []
        model.eval()
        with torch.no_grad():
            for idx, (image, label, _) in enumerate(dataset):
                image = image.unsqueeze(0).cuda()  # add batch dim
                label = torch.tensor(label).cuda()

                output = model(image).logits
                pred = output.argmax(dim=1)

                if pred.item() != label.item():
                    failure_indices.append(idx)

        # Save to pickle
        with open(failure_path, 'wb') as f:
            pickle.dump(failure_indices, f)

    # Create filtered failure-case dataset
    return failure_indices

def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr

class ColoredObject(Dataset):
    def __init__(self, root_dir, env, mode='train', ratio=None, transforms=None, processor=None):
        self.root_dir = root_dir
        self.env = env
        self.env_dir = os.path.join(root_dir, env)
        self.transforms = transforms
        self.processor = processor
        self.mode = mode

        metadata_path = os.path.join(self.env_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        if ratio is not None:
            total_samples = len(self.metadata)
            num_samples = int(ratio * total_samples)
            indices = torch.randperm(total_samples)[:num_samples]
            self.metadata = [self.metadata[indice] for indice in indices]

    def __len__(self):
        return len(self.metadata)

    def combine(self, dataset):
        self.metadata.extend(dataset.metadata)
        self.env = 'all_train'

    def set_transforms(self, transforms):
        self.transforms = transforms

    def set_processor(self, processor):
        self.processor = processor

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        image_path = os.path.join(self.root_dir, entry['file_name'])
        img = Image.open(image_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        if self.processor:
            img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        label = entry['label']
        bg_color = tuple(entry['bg_color'])
        if self.mode == 'bg':
            ctft_img_path = os.path.join(self.root_dir, entry['ctft_bg'])
            ctft_img = Image.open(ctft_img_path).convert("RGB")
            ctft_bg_color = entry['ctft_bg_color']
            if self.transforms:
                ctft_img = self.transforms(ctft_img)
            if self.processor:
                ctft_img = self.processor(images=ctft_img, return_tensors="pt")["pixel_values"].squeeze(0)
            return img, ctft_img, label, bg_color
        elif self.mode == 'fg':
            ctft_img_path = os.path.join(self.root_dir, entry['ctft_fg'])
            ctft_img = Image.open(ctft_img_path).convert("RGB")
            if self.transforms:
                ctft_img = self.transforms(ctft_img)
            if self.processor:
                ctft_img = self.processor(images=ctft_img, return_tensors="pt")["pixel_values"].squeeze(0)
            return img, ctft_img, label, bg_color, ctft_bg_color

        return img, label, bg_color

class SceneObject(Dataset):
    def __init__(self, root_dir, env, ratio=None, transforms=None, processor=None):
        self.root_dir = root_dir
        self.env = env
        self.env_dir = os.path.join(root_dir, env)
        self.transforms = transforms
        self.processor = processor

        metadata_path = os.path.join(self.env_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        if ratio is not None:
            total_samples = len(self.metadata)
            num_samples = int(ratio * total_samples)
            indices = torch.randperm(total_samples)[:num_samples]
            self.metadata = [self.metadata[indice] for indice in indices]

    def __len__(self):
        return len(self.metadata)

    def combine(self, dataset):
        self.metadata.extend(dataset.metadata)
        self.env = 'all_train'

    def set_transforms(self, transforms):
        self.transforms = transforms

    def set_processor(self, processor):
        self.processor = processor

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        image_path = os.path.join(self.root_dir, entry['file_name'])
        img = Image.open(image_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        if self.processor:
            img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        label = entry['label']
        bg_color = tuple(entry['bg_scene'])
        return img, label, bg_color

class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None, select_class=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      with open(os.path.join(self.root, 'ColoredMNIST', env) + '.p', 'rb') as file:
        self.data_label_tuples = pickle.load(file)
    elif env == 'all_train':
      with open(os.path.join(self.root, 'ColoredMNIST', 'train1.p'), 'rb') as file:
        train1 = pickle.load(file)
      with open(os.path.join(self.root, 'ColoredMNIST', 'train2.p'), 'rb') as file:
        train2 = pickle.load(file)
      self.data_label_tuples = train1 + train2
      if select_class is not None and select_class != 'all':
        filtered = [(img, label) for img, label in self.data_label_tuples if label == select_class]
        self.data_label_tuples = filtered
    elif env == 'all_train_unbiased':
      self.prepare_colored_mnist_unbiased()
      with open(os.path.join(self.root, 'ColoredMNIST', 'train1_unbiased.p'), 'rb') as file:
        train1 = pickle.load(file)
      with open(os.path.join(self.root, 'ColoredMNIST', 'train2_unbiased.p'), 'rb') as file:
        train2 = pickle.load(file)
      self.data_label_tuples = train1 + train2
      if select_class is not None and select_class != 'all':
        filtered = [(img, label) for img, label in self.data_label_tuples if label == select_class]
        self.data_label_tuples = filtered
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.p')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      # if np.random.uniform() < 0.25:
      #   binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    os.makedirs(colored_mnist_dir, exist_ok=True)
    with open(os.path.join(colored_mnist_dir, 'train1.p'), 'wb') as file:
      pickle.dump(train1_set, file)
    with open(os.path.join(colored_mnist_dir, 'train2.p'), 'wb') as file:
      pickle.dump(train2_set, file)
    with open(os.path.join(colored_mnist_dir, 'test.p'), 'wb') as file:
      pickle.dump(test_set, file)

  def prepare_colored_mnist_unbiased(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1_unbiased.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2_unbiased.p')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test_unbiased.p')):
      print('unbiased Colored MNIST dataset already exists')
      return

    print('Preparing unbiased Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      # if np.random.uniform() < 0.25:
      #   binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.5:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.5:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.5:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    os.makedirs(colored_mnist_dir, exist_ok=True)
    with open(os.path.join(colored_mnist_dir, 'train1_unbiased.p'), 'wb') as file:
      pickle.dump(train1_set, file)
    with open(os.path.join(colored_mnist_dir, 'train2_unbiased.p'), 'wb') as file:
      pickle.dump(train2_set, file)
    with open(os.path.join(colored_mnist_dir, 'test_unbiased.p'), 'wb') as file:
      pickle.dump(test_set, file)

class WaterbirdDataset(Dataset):
  def __init__(self, data_correlation, split, root_dir, transform, worst_group=None, subset_fraction=1.0, seed=42):
    self.split_dict = {
      'train': 0,
      'val': 1,
      'test': 2
    }
    self.env_dict = {
      (0, 0): 0,
      (0, 1): 1,
      (1, 0): 2,
      (1, 1): 3
    }
    self.split = split
    self.root_dir = root_dir
    self.dataset_name = "waterbird_complete" + "{:0.2f}".format(data_correlation)[-2:] + "_forest2water2"
    self.dataset_dir = os.path.join(self.root_dir, self.dataset_name)
    if not os.path.exists(self.dataset_dir):
      raise ValueError(
        f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
    self.metadata_df = pd.read_csv(
      os.path.join(self.dataset_dir, 'metadata.csv'))
    self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]
    if worst_group:
      self.metadata_df = self.metadata_df[
        (self.metadata_df['y'] == 1) &
        (self.metadata_df['place'] == 0)
      ]

    if 0 < subset_fraction < 1.0:
        self.metadata_df = self.metadata_df.sample(frac=subset_fraction, random_state=seed).reset_index(drop=True)

    self.y_array = self.metadata_df['y'].values
    self.place_array = self.metadata_df['place'].values
    self.filename_array = self.metadata_df['img_filename'].values
    self.transform = transform
    self.target_name = 'y_array'
    self.env_array = np.array([self.env_dict[(y, place)] for y, place in zip(self.y_array, self.place_array)])

  def __len__(self):
    return len(self.filename_array)

  def get_group(self, group_id):
    wg_idx = np.where(self.env_array == group_id)[0]
    self.y_array = self.y_array[wg_idx]
    self.place_array = self.place_array[wg_idx]
    self.filename_array = self.filename_array[wg_idx]

  def filter_from_list(self, filter_list):
    self.y_array = self.y_array[filter_list]
    self.place_array = self.place_array[filter_list]
    self.filename_array = self.filename_array[filter_list]

  def __getitem__(self, idx):
    y = self.y_array[idx]
    place = self.place_array[idx]
    img_filename = os.path.join(
      self.dataset_dir,
      self.filename_array[idx])
    img = Image.open(img_filename).convert('RGB')
    img = self.transform(img)

    return img, y, place

class ConsistencyDataset(Dataset):
  def __init__(self, transform, root_dir, worst_group=None, split=None, label_select=None):
    self.envs = ['water', 'land']
    self.env_dict = {
      (0, 0): 0,
      (0, 1): 1,
      (1, 0): 2,
      (1, 1): 3
    }
    self.root_dir = root_dir
    self.split = split
    self.split_dict = {
      'train': 0,
      'val': 1,
      'test': 2
    }
    self.dataset_name = [f'{env}_bg' for env in self.envs]
    self.dataset_dir = [os.path.join(self.root_dir, dataset_name, 'images') for dataset_name in self.dataset_name]
    if not os.path.exists(self.dataset_dir[0]):
      raise ValueError(
        f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
    self.metadata_df = pd.read_csv(
      os.path.join(self.root_dir, 'metadata.csv'))
    if self.split is not None:
      self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]
    if label_select is not None:
      self.metadata_df = self.metadata_df[(self.metadata_df['y'] == label_select)]
    if worst_group:
      self.metadata_df = self.metadata_df[
        (self.metadata_df['y'] == 1) &
        (self.metadata_df['place'] == 0)
      ]
    self.y_array = self.metadata_df['y'].values
    self.place_array = self.metadata_df['place'].values
    self.filename_array = self.metadata_df['img_filename'].values
    self.transform = transform

  def __len__(self):
    return len(self.filename_array)

  def __getitem__(self, idx):
    out_img = []
    y = self.y_array[idx]
    place = self.place_array[idx]
    for env_dir in self.dataset_dir:
      img_filename = os.path.join(
        env_dir,
        self.filename_array[idx])
      img = Image.open(img_filename).convert('RGB')
      img = self.transform(img)
      out_img.append(img)

    return out_img, y, place

class TerraIncognita(Dataset):
    def __init__(self, root_dir, split, environment, transform=None):
        self.root_dir = root_dir  # e.g., "data/PACS/sketch"
        self.transform = transform

        path = os.path.join(root_dir, environment)
        env_dataset = ImageFolder(path,
                                  transform=transform)
        split_path = os.path.join(root_dir, f'{environment}_split.p')

        if split is not None:
            with open(split_path, 'rb') as file:
                if '38' in environment:
                    saved_val_indices, saved_train_indices = pickle.load(file)
                else:
                    saved_train_indices, saved_val_indices = pickle.load(file)

        if split == 'val':
            self.datasets = Subset(env_dataset, saved_val_indices)
        elif split == 'train':
            self.datasets = Subset(env_dataset, saved_train_indices)
        else:
            self.datasets = env_dataset

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]

class TFDSVisionDataset(Dataset):
    def __init__(self, tfds_dataset, transform):
        self.data = list(tfds_dataset)   # materialize once
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        image = example[0].numpy()
        label = int(example[1].numpy())

        image = Image.fromarray(image).convert("RGB")
        image = self.transform(image)

        return image, label

def get_metashift_dataset(data_label_correlation=None,  # not used but kept for API consistency
                          split="train",
                          transform=None,
                          root_dir="/home/yxpengcs/PycharmProjects/Moon-Shape-ICML-2023/experiments/metashift/data",
                          val_size=0.2,
                          seed=42):
    """
    Returns a dataset (train, val, or test) similar to get_waterbird_dataset.
    For 'train' and 'val', split from 'train' metadata entries.
    """
    # Load all "train" entries if split is train/val
    full_dataset = MetaShiftDataset(split='train' if split in ['train', 'val'] else split,
                                    root_dir=root_dir,
                                    transform=transform)

    if split == 'test':
        return full_dataset

    # Stratified split based on class + env
    metadata = full_dataset.samples
    stratify_col = metadata["class"].astype(str) + "_" + metadata["env"].astype(str)
    train_idx, val_idx = train_test_split(
        metadata.index,
        test_size=val_size,
        stratify=stratify_col,
        random_state=seed,
    )
    if split == "train":
        return Subset(full_dataset, train_idx)
    elif split == "val":
        return Subset(full_dataset, val_idx)
class MetaShiftDataset(Dataset):
    def __init__(self, split, root_dir, transform=None):
        """
        Args:
            split (str): One of ['train', 'majority-val', 'minority-val']
            root_dir (str): Path to 'data' directory (e.g., '../../experiments/metashift/data')
            transform (callable, optional): Transform to apply to each image
        """
        self.split = split
        self.root_dir = root_dir
        self.transform = transform

        # Load metadata
        metadata_path = os.path.join(root_dir, "metadata.csv")
        metadata_df = pd.read_csv(metadata_path)
        if split != 'train':
            self.samples = metadata_df[metadata_df["split"] != 'train'].reset_index(drop=True)
        else:
            self.samples = metadata_df[metadata_df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row['class'])  # 0 = cat, 1 = dog
        env = int(row['env'])
        env = 0 if env == 0 or env == 2 else 1 # 0 = indoor, 1 = outdoor

        return img, label, env

class CutoutTransform:
    """Random erasing (cutout) on image."""
    def __init__(self, size=32, p=1.0):
        self.size = size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        img_np[y:y+self.size, x:x+self.size] = 0
        return Image.fromarray(img_np)

class StyleModelTransform:
    def __init__(self, model, device="cuda", p=1.0):
        """
        model: a pretrained style transfer model (expects tensor input)
        device: 'cuda' or 'cpu'
        p: probability to apply stylization
        """
        self.model = model.to(device).eval()
        self.device = device
        self.p = p

        # preprocessing / postprocessing
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.denormalize = lambda t: (t * torch.tensor([0.229,0.224,0.225], device=device)[:,None,None]
                                        + torch.tensor([0.485,0.456,0.406], device=device)[:,None,None]).clamp(0,1)

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() > self.p:
            return img

        x = self.to_tensor(img).unsqueeze(0).to(self.device)
        x = self.normalize(x)

        with torch.no_grad():
            y = self.model(x)

        y = self.denormalize(y.squeeze(0)).cpu()
        return self.to_pil(y)

class ContinuousCorruptionTransform:
    def __init__(self, corruption_type, intensity):
        """
        corruption_type: str, e.g., 'gaussian_noise'
        intensity: float, continuous severity parameter
        """
        self.corruption_type = corruption_type
        self.intensity = intensity

    def __call__(self, x):
        x = torchvision.transforms.ToTensor()(x)
        if self.corruption_type == 'gaussian_noise':
            return self.gaussian_noise(x, self.intensity)
        elif self.corruption_type == 'shot_noise':
            return self.shot_noise(x, self.intensity)
        elif self.corruption_type == 'motion_blur':
            return self.motion_blur(x, self.intensity)
        else:
            return x

    def gaussian_noise(self, x, sigma):
        return (x + sigma * torch.randn_like(x)).clamp(0, 1)

    def shot_noise(self, x, lam):
        # lam is continuous noise level
        return torch.poisson(x * lam) / lam

    def motion_blur(self, x, alpha):
        # alpha in [0, 1]: blend identity with blur kernel
        kernel = get_motion_blur_kernel(alpha)  # define yourself
        return convolve_tensor(x, kernel)

def get_loader(task, model):
    config = timm.data.resolve_model_data_config(model)
    transform_train = timm.data.create_transform(**config, is_training=True)
    transform_test = timm.data.create_transform(**config, is_training=False)

    if task == 'cifar10':
        cifar_c = ["fog", "frost", "motion_blur", "brightness", "defocus_blur", "snow", "zoom_blur"]
        severities = [1, 2, 3, 4, 5]
        trainset = torchvision.datasets.CIFAR10(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10", train=True, download=True, \
                                     transform=transform_train)

        valset = torchvision.datasets.CIFAR10(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10", train=False, download=True, \
                                                 transform=transform_test)

        testset = {}
        for data in cifar_c:
            for severity in severities:
                testset[f"{data}-{severity}"] = CIFAR10_C(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10/CIFAR-10-C/", data_type=data, severity=severity,
                                    transform=transform_test)

    if task == "waterbirds":
        trainset = get_waterbird_dataset(data_label_correlation = 0.95,
                        split="train", transform = transform_train, root_dir = '../vit-spurious-robustness-modified/datasets')

        valset = get_waterbird_dataset(data_label_correlation = 0.95,
                        split="val", transform = transform_test,root_dir = '../vit-spurious-robustness-modified/datasets')
        testset = get_waterbird_dataset(data_label_correlation=0.95,
                                        split="test", transform=transform_test,
                                        root_dir='../vit-spurious-robustness-modified/datasets')

    if task == 'metashift':
        trainset = get_metashift_dataset(split="train", transform=transform_train)
        valset = get_metashift_dataset(split="val", transform=transform_test)
        testset = get_metashift_dataset(split="test", transform=transform_test)

    if task == 'metashift-control':
        trainset = get_metashift_dataset(split="train", root_dir=f'/home/yxpengcs/PycharmProjects/Moon-Shape-ICML-2023/experiments/metashift-0.1/data', transform=transform_train)
        valset = get_metashift_dataset(split="val", root_dir=f'/home/yxpengcs/PycharmProjects/Moon-Shape-ICML-2023/experiments/metashift-0.1/data', transform=transform_test)
        testset = {}
        for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            testset[f'{i}'] = get_metashift_dataset(split="test", root_dir=f'/home/yxpengcs/PycharmProjects/Moon-Shape-ICML-2023/experiments/metashift-{i}/data', transform=transform_test)

    if task == 'yearbook':
        import argparse
        from wildtime import dataloader
        config = {'dataset': 'yearbook', 'regression': False, 'prediction_type': None, 'method': 'erm', 'device': 0,
                  'random_seed': 1, 'train_update_iter': 3000, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0,
                  'mini_batch_size': 32, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False,
                  'split_time': 1970, 'eval_next_timestamps': 10, 'load_model': False, 'eval_all_timestamps': False,
                  'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0,
                  'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 10, 'group_size': 5, 'non_overlapping': False,
                  'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False,
                  'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0,
                  'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 300, 'data_dir': './datasets',
                  'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
        configs = argparse.Namespace(**config)
        whole_dataset = dataloader.getdata(configs)
        trainset = Yearbook(whole_dataset, 'train', transform=transform_train)
        valset = Yearbook(whole_dataset, 'val', transform=transform_test)
        testset = Yearbook(whole_dataset, 'test', transform=transform_test)

    if task == 'camelyon17-hospital1':
        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            split_scheme="official"
        )
        dataset = whole_dataset.get_subset("val", transform=transform_test)
        torch.manual_seed(0)
        indices = torch.randperm(len(dataset))[:4000]
        dataset = Subset(dataset, indices)

    if task == 'camelyon17-id':
        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            split_scheme="official"
        )
        dataset = whole_dataset.get_subset("id_val", transform=transform_test)

    if task == 'camelyon17-hospital2':
        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            split_scheme="official"
        )
        dataset = whole_dataset.get_subset("test", transform=transform_test)
        torch.manual_seed(0)
        indices = torch.randperm(len(dataset))[:4000]
        dataset = Subset(dataset, indices)

    if task == 'fmow':
        from wilds.datasets.wilds_dataset import WILDSSubset
        import pandas as pd
        _old_to_datetime = pd.to_datetime

        def _patched_to_datetime(arg, *a, **kw):
            return _old_to_datetime(arg, format="ISO8601", utc=True)

        pd.to_datetime = _patched_to_datetime
        whole_dataset = wilds.get_dataset(
            dataset="fmow",
            root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            download=True,
        )
        trainset = whole_dataset.get_subset('train', transform=transform_train)
        testset = {}

        valset = whole_dataset.get_subset('id_val', transform=transform_test)
        testsetv2 = whole_dataset.get_subset('val', transform=transform_test)
        testsetv3 = whole_dataset.get_subset('test', transform=transform_test)

        def get_groups(testset):
            groups = whole_dataset._eval_groupers['region'].metadata_to_group(testset.metadata_array)
            group_testset = []
            for i in range(5):
                idx = np.where(groups == i)[0]
                group_testset.append(WILDSSubset(testset, idx, None))

            return group_testset

        testsetv2_groups = get_groups(testsetv2)
        testsetv3_groups = get_groups(testsetv3)
        for id, group in enumerate(testsetv2_groups):
            testset[f'time1_region{id}'] = group
        for id, group in enumerate(testsetv3_groups):
            testset[f'time2_region{id}'] = group

    if task == 'iwildcam':
        whole_dataset = wilds.get_dataset(
            dataset="iwildcam",
            root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            download=True,
        )
        trainset = whole_dataset.get_subset("train", transform=transform_train)
        valset = whole_dataset.get_subset("val", transform=transform_test)

        # Get test set (split = 'test')
        full_test_set = whole_dataset.get_subset("test", transform=transform_test)

        # Get metadata (e.g., location)
        metadata = full_test_set.metadata_array  # shape (N, D)
        location_col = whole_dataset.metadata_fields.index("location")

        # List of unique OOD domains (locations)
        unique_locations = metadata[:, location_col].unique()

        # Make a dictionary of test sets per location
        testset = {}
        torch.manual_seed(0)  # ensure reproducibility of sampling

        for loc in unique_locations:
            loc_indices = (metadata[:, location_col] == loc).nonzero(as_tuple=True)[0]
            subset = Subset(full_test_set, loc_indices)
            testset[f"location_{int(loc.item())}"] = subset

    if task == 'PACS-cartoon':
        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', 'test',  transform=transform_test)

    if task == 'PACS-art_painting':
        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', 'test,', transform=transform_test)

    if task == 'PACS-photo':
        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/photo/split.json', 'test', transform=transform_test)

    if task == 'PACS-sketch':
        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'test', transform=transform_test)

    if task == 'terra-incognita-38-location46':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_46", transform=transform_test)

    if task == 'terra-incognita-38-location43':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_43", transform=transform_test)

    if task == 'terra-incognita-38-location100':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_100", transform=transform_test)

    if task == 'terra-incognita-38-location38':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_38", transform=transform_test)

    if task == 'terra-incognita-43-location38':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_38", transform=transform_test)

    if task == 'terra-incognita-43-location43':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_43", transform=transform_test)

    if task == 'terra-incognita-43-location46':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_46", transform=transform_test)

    if task == 'terra-incognita-43-location100':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_100", transform=transform_test)

    if task == 'terra-incognita-46-location38':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_38", transform=transform_test)

    if task == 'terra-incognita-46-location43':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_43", transform=transform_test)

    if task == 'terra-incognita-46-location46':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_46", transform=transform_test)

    if task == 'terra-incognita-46-location100':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_100", transform=transform_test)

    if task == 'terra-incognita-100-location38':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_38", transform=transform_test)

    if task == 'terra-incognita-100-location43':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_43", transform=transform_test)

    if task == 'terra-incognita-100-location46':
        root_dir = '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita'
        dataset = TerraIncognita(root_dir, 'val', "location_46", transform=transform_test)

    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    # val_sampler = SequentialSampler(valset)
    # test_sampler = SequentialSampler(testset)
    loader = DataLoader(dataset,
                             # sampler=val_sampler,
                             batch_size=256,
                             num_workers=4,
                             pin_memory=False,persistent_workers=False, prefetch_factor=2)
    return loader

def setup_ood_datasets(dataset_name, transform=None, split='val', model_name=None, num_examples=None, fragment=None, class_ranges=None, device='cuda'):
    if not transform:
        transform = get_transform(model_dict[model_name])
    if dataset_name == 'PACS-mean-sketch':

        test_datasets = [
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/photo/split.json', split,
                              transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', split,
                        transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', split,
                        transform=transform),
        ]
    elif dataset_name == 'PACS-mean-photo':

        test_datasets = [
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', split,
                              transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', split,
                        transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', split,
                        transform=transform),
        ]
    elif dataset_name == 'PACS-mean-cartoon':

        test_datasets = [
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/photo/split.json', split,
                              transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', split,
                        transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', split,
                        transform=transform),
        ]
    elif dataset_name == 'PACS-mean-art_painting':

        test_datasets = [
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/photo/split.json', split,
                              transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', split,
                        transform=transform),
            PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', split,
                        transform=transform),
        ]

    elif dataset_name in ('PACS-full-photo', 'PACS-full-sketch', 'PACS-full-cartoon', 'PACS-full-art_painting'):
        from torch.utils.data import ConcatDataset as TorchConcatDataset
        _pacs_base = '/home/yxpengcs/PycharmProjects/assaying-ood/webdata'
        _all_domains = {'photo': f'{_pacs_base}/photo/split.json',
                        'sketch': f'{_pacs_base}/sketch/split.json',
                        'cartoon': f'{_pacs_base}/cartoon/split.json',
                        'art_painting': f'{_pacs_base}/art_painting/split.json'}
        _source = dataset_name.split('PACS-full-')[1]
        _indiv = [PACSDataset(path, 'all', transform=transform)
                  for dom, path in _all_domains.items() if dom != _source]
        _combined = TorchConcatDataset(_indiv)
        test_datasets = _indiv + [_combined]

    return [VisionEAPDataset(dataset, task=dataset_name, num_examples=num_examples, model_name=model_name, device=device, fragment=fragment) for dataset in test_datasets]

def setup_dataset(dataset_name, transform=None, split='val', model_name=None, num_examples=None, fragment=None, class_ranges=None, device='cuda'):
    ood_dataset = None
    if not transform:
        transform = get_transform(model_dict[model_name], train=split=='train')
    if 'IN-set2' in dataset_name:

        if 'v2-MF' in dataset_name:
            import tensorflow_datasets as tfds
            dataset = TFDSVisionDataset(tfds.load("imagenet_v2", split="test", as_supervised=True),
                                                          transform=transform)
        elif 'id-200' in dataset_name:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenet-val",
                transform=transform
            )
            dataset_r = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenet-r",
                transform=transform
            )
            # 0–199 → synset
            r_synsets = set(dataset_r.classes)  # 200 synsets
            val_classes = dataset.classes  # list of 1000 synsets
            val_class_to_idx = dataset.class_to_idx  # synset -> idx
            filtered_samples = [
                (path, label)
                for path, label in dataset.samples
                if val_classes[label] in r_synsets
            ]
            dataset.samples = filtered_samples
            dataset.targets = [label for _, label in filtered_samples]

        elif 'id' in dataset_name:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenet-val",
                transform=transform
            )
        elif 'v2-top' in dataset_name:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenetv2-top-images-format-val",
                transform=transform
            )
        elif 'v2-threshold0.7' in dataset_name:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenetv2-threshold0.7-format-val",
                transform=transform
            )
        elif 'imagenet-r' in dataset_name:
            # ImageNet-R (renditions); folder: imagenet-r
            with open("../vit-spurious-robustness/imagenet_class_index.json") as f:
                class_idx = json.load(f)

            synset_to_1k = {v[0]: int(k) for k, v in class_idx.items()}
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenet-r",
                transform=transform
            )
            # 0–199 → synset
            idx_to_synset = dataset.classes

            # 0–199 → 0–999
            idx_to_1k = [synset_to_1k[s] for s in idx_to_synset]

            # Modify samples
            dataset.samples = [
                (path, idx_to_1k[label])
                for path, label in dataset.samples
            ]

            # Modify targets (important!)
            dataset.targets = [idx_to_1k[label] for label in dataset.targets]
            torch.manual_seed(0)
            indices = torch.randperm(len(dataset))[:10].tolist()
            dataset = Subset(dataset, indices)
        elif 'imagenet-a' in dataset_name:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenet-a",
                transform=transform
            )
        elif 'imagenet-s' in dataset_name:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root="/home/yxpengcs/Datasets/imagenet/imagenet-sketch",
                transform=transform
            )
            torch.manual_seed(0)
            indices = torch.randperm(len(dataset))[:10000].tolist()
            dataset = Subset(dataset, indices)
        elif 'imagenet-c' in dataset_name:
            # Expected format: IN-set2-*-imagenet-c-<corruption>-<severity>
            # Example: IN-set2-mean-imagenet-c-gaussian_noise-3
            from torchvision.datasets import ImageFolder
            parts = dataset_name.split('-')
            try:
                idx = parts.index('c')
                if idx < 1 or parts[idx - 1].lower() != 'imagenet':
                    raise ValueError("imagenet-c not found")
                corruption = parts[idx + 1]
                severity = int(parts[idx + 2])
            except (ValueError, IndexError, AssertionError) as e:
                raise ValueError(
                    f"IN-set2 ImageNet-C: expect imagenet-c-<corruption>-<severity> "
                    f"(e.g. IN-set2-mean-imagenet-c-gaussian_noise-3), got {dataset_name!r}"
                ) from e
            inc_root = os.path.join('/home/yxpengcs/Datasets/imagenet/', 'imagenet-c', corruption, str(severity))
            if not os.path.isdir(inc_root):
                raise FileNotFoundError(
                    f"ImageNet-C path not found: {inc_root}. "
                    "Ensure imagenet-c is at <imagenet_root>/imagenet-c/<corruption>/<severity>/."
                )

            def _pil_loader_rgb(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')

            dataset = ImageFolder(root=inc_root, transform=transform, loader=_pil_loader_rgb)
            # if num_examples is not None:
            torch.manual_seed(0)
            indices = torch.randperm(len(dataset))[:10000].tolist()
            dataset = Subset(dataset, indices)
        else:
            raise ValueError(
                f"IN-set2: unrecognized variant in {dataset_name!r}. "
                "Use v2, imagenet-r, or imagenet-c (e.g. IN-set2-mean-imagenet-c-gaussian_noise-3)."
            )
        intervention_dataset = dataset

    elif 'IN' in dataset_name:
        if dataset_name == 'IN-dog':
            class_ranges = [208, 212, 263, 189, 245]
            ctft_class_ranges = [407, 656, 436, 468, 511]
        elif dataset_name == 'IN-car':
            class_ranges = [407, 656, 436, 468, 511]
            ctft_class_ranges = [208, 212, 263, 189, 245]
        transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=224),
            transforms.ToTensor()
        ])
        split = 'train' if split == 'train' else 'val'
        if model_name == 'ViT-B_16':
            dataset = ImageNetDataset(root_dir=f'/home/yxpengcs/Datasets/imagenet/{split}',
                                      processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"),
                                      select_class=class_ranges, ctft_class_ranges=ctft_class_ranges)
        else:
            dataset = ImageNetDataset(root_dir=f'/home/yxpengcs/Datasets/imagenet/{split}',
                                      transform=transform, select_class=class_ranges, ctft_class_ranges=ctft_class_ranges)
    elif dataset_name == 'PACS-mean' or dataset_name == 'PACS-mean-sketch' or dataset_name == 'PACS-photo-mean-sketch' or dataset_name == 'PACS-cartoon-mean-sketch' or dataset_name == 'PACS-art_painting-mean-sketch':

        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', split, transform=transform)
        intervention_dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', split, transform=transform)
    elif dataset_name == 'PACS-mean-photo' or dataset_name == 'PACS-photo-mean' or dataset_name == 'PACS-cartoon-mean-photo' or dataset_name == 'PACS-art_painting-mean-photo':

        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/photo/split.json', split, transform=transform)
        intervention_dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/photo/split.json', split, transform=transform)
    elif dataset_name == 'PACS-mean-cartoon' or dataset_name == 'PACS-photo-mean-cartoon' or dataset_name == 'PACS-cartoon-mean' or dataset_name == 'PACS-art_painting-mean-cartoon':

        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', split, transform=transform)
        intervention_dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/cartoon/split.json', split, transform=transform)
    elif dataset_name == 'PACS-mean-art_painting' or dataset_name == 'PACS-photo-mean-art_painting' or dataset_name == 'PACS-cartoon-mean-art_painting' or dataset_name == 'PACS-art_painting-mean':

        dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', split, transform=transform)
        intervention_dataset = PACSDataset('/home/yxpengcs/PycharmProjects/assaying-ood/webdata/art_painting/split.json', split, transform=transform)
    elif 'PACS-set2' in dataset_name:

        if 'id1' in dataset_name:
            dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'test', transform=transform)
            intervention_dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'test', transform=transform)
        elif 'id2' in dataset_name:
            dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'train', transform=transform)
            intervention_dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'train', transform=transform)
        else:
            domain = dataset_name.split('-')[3]
            split = dataset_name.split('-')[4]
            if domain not in ['train', 'val', 'test']:
                tfm = copy.deepcopy(transform)
                tfm.transforms.insert(1, CorruptionTransform(domain, int(split)))
                dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'val',
                                      transform=tfm)
                intervention_dataset = PACSDataset(
                    f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/sketch/split.json', 'val',
                    transform=tfm)
            else:
                dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/{domain}/split.json', split, transform=transform)
                intervention_dataset = PACSDataset(f'/home/yxpengcs/PycharmProjects/assaying-ood/webdata/{domain}/split.json', split, transform=transform)
    elif 'camelyon17-set2' in dataset_name:
        from wilds.datasets.wilds_dataset import WILDSSubset

        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/", split_scheme="official"
        )
        with open('/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/camelyon17_v1.0/splits.pkl', 'rb') as file:
            train_idx, val_idx, testsets_idx = pickle.load(file)
        with open('/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/camelyon17_v1.0/splits_id.pkl', 'rb') as file:
            id_idx = pickle.load(file)

        if '-id' in dataset_name:
            dataset = WILDSSubset(whole_dataset, indices=list(val_idx), transform=transform)
            intervention_dataset = WILDSSubset(whole_dataset, indices=list(val_idx), transform=transform)
        elif 'corrupt' in dataset_name:
            currupt_type = dataset_name.split('-')[3]
            severity = dataset_name.split('-')[4]
            tfm = copy.deepcopy(transform)
            tfm.transforms.insert(1, CorruptionTransform(currupt_type, int(severity)))
            dataset = WILDSSubset(whole_dataset, indices=list(val_idx)[:400], transform=tfm)
            intervention_dataset = WILDSSubset(whole_dataset, indices=list(val_idx)[:400], transform=tfm)
        elif 'transform' in dataset_name:
            transform_type = dataset_name.split('-')[3].replace('_transform', '')
            tfm = copy.deepcopy(transform)
            if transform_type == 'random_crop_resize':
                severe_crop_resize = torchvision.transforms.RandomResizedCrop(
                    size=(224, 224),  # output size (same as model input)
                    scale=(0.2, 0.4),  # crop between 20% and 40% of original area
                    ratio=(0.5, 2.0),  # extreme aspect ratios for distortion
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )
                tfm.transforms.insert(1, severe_crop_resize)
            elif transform_type == 'cutout':
                cutout = CutoutTransform(size=80, p=1.0)
                tfm.transforms.insert(1, cutout)
            elif transform_type == 'countour_stylize':
                countour = CountourStylizeTransform(p=1.0)
                tfm.transforms.insert(1, countour)
            elif transform_type == 'emboss_stylize':
                emboss = EmbossStylizeTransform(p=1.0)
                tfm.transforms.insert(1, emboss)
            elif transform_type == 'edge_enhance_stylize':
                edge_enhance = EdgeEnhanceStylizeTransform(p=1.0)
                tfm.transforms.insert(1, edge_enhance)
            elif transform_type == 'edge_stylize':
                edge_enhance = EdgeStylizeTransform()
                tfm.transforms.insert(1, edge_enhance)
            elif transform_type == 'posterize_stylize':
                edge_enhance = PosterizeStylizeTransform()
                tfm.transforms.insert(1, edge_enhance)
            elif transform_type == 'solarize_stylize':
                edge_enhance = SolarizeStylizeTransform()
                tfm.transforms.insert(1, edge_enhance)
            elif transform_type == 'palette_stylize':
                edge_enhance = PaletteStylizeTransform()
                tfm.transforms.insert(1, edge_enhance)
            elif transform_type == 'cartoon_stylize':
                edge_enhance = CartoonStylizeTransform()
                tfm.transforms.insert(1, edge_enhance)
            dataset = WILDSSubset(whole_dataset, indices=list(val_idx)[:400], transform=tfm)
            intervention_dataset = WILDSSubset(whole_dataset, indices=list(val_idx)[:400], transform=tfm)
        elif 'sensitive' in dataset_name:
            currupt_type = dataset_name.split('-')[3]
            severity = dataset_name.split('-')[4]
            tfm = copy.deepcopy(transform)
            tfm.transforms.insert(1, ContinuousCorruptionTransform(currupt_type, float(severity)))
            dataset = WILDSSubset(whole_dataset, indices=list(val_idx)[:400], transform=tfm)
            intervention_dataset = WILDSSubset(whole_dataset, indices=list(val_idx)[:400], transform=tfm)
        else:
            hospital = dataset_name.split('-')[3].split('_')[0]
            slide = dataset_name.split('-')[3].split('_')[1]
            try:
                idx = testsets_idx[f'{hospital}_{slide}']
            except:
                idx = id_idx[f'{hospital}_{slide}']
            dataset = WILDSSubset(whole_dataset, indices=list(idx), transform=transform)
            intervention_dataset = WILDSSubset(whole_dataset, indices=list(idx), transform=transform)
    elif dataset_name == 'camelyon17-mean':

        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/", split_scheme="official"
        )
        dataset = whole_dataset.get_subset("id_val", transform=transform)
        torch.manual_seed(0)
        indices = torch.randperm(len(dataset))[:1600]
        intervention_dataset = whole_dataset.get_subset("id_val", transform=transform)
        dataset = Subset(dataset, indices)
        intervention_dataset = Subset(intervention_dataset, indices)
    elif dataset_name == 'camelyon17-mean-hospital1':

        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/", split_scheme="official"
        )
        dataset = whole_dataset.get_subset("val", transform=transform)
        torch.manual_seed(0)
        indices = torch.randperm(len(dataset))[:1600]
        intervention_dataset = whole_dataset.get_subset("val", transform=transform)
        dataset = Subset(dataset, indices)
        intervention_dataset = Subset(intervention_dataset, indices)
    elif 'cifar' in dataset_name:

        type = dataset_name.split('-')[2]
        if type == 'id':
            dataset = torchvision.datasets.CIFAR10(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10", train=False, download=True, \
                                                 transform=transform)
            intervention_dataset = torchvision.datasets.CIFAR10(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10",
                                                   train=False, download=True, \
                                                   transform=transform)
        else:
            severity = int(dataset_name.split('-')[3])
            dataset = CIFAR10_C(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10/CIFAR-10-C/", data_type=type, severity=severity,
                                        transform=transform)
            intervention_dataset = CIFAR10_C(root=f"/home/yxpengcs/PycharmProjects/ATC_code/data/CIFAR10/CIFAR-10-C/", data_type=type, severity=severity,
                                        transform=transform)
        # torch.manual_seed(0)
        # indices = torch.randperm(len(dataset))[:1600]
        # dataset = Subset(dataset, indices)
        # intervention_dataset = Subset(intervention_dataset, indices)
    elif 'iwildcam' in dataset_name:

        whole_dataset = wilds.get_dataset(
            dataset="iwildcam",
            root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            download=True,
        )
        if 'id' in dataset_name:
            dataset = whole_dataset.get_subset("val", transform=transform)
            intervention_dataset = whole_dataset.get_subset("val", transform=transform)
        else:
            loc_num = int(dataset_name.split('-')[-1])
            # Get test set (split = 'test')
            full_test_set = whole_dataset.get_subset("test", transform=transform)

            # Get metadata (e.g., location)
            metadata = full_test_set.metadata_array  # shape (N, D)
            location_col = whole_dataset.metadata_fields.index("location")

            # List of unique OOD domains (locations)
            loc_indices = (metadata[:, location_col] == loc_num).nonzero(as_tuple=True)[0]

            dataset = Subset(full_test_set, loc_indices)
            intervention_dataset = Subset(full_test_set, loc_indices)
    elif 'fmow' in dataset_name:

        time = dataset_name.split('-')[2]
        from wilds.datasets.wilds_dataset import WILDSSubset
        import pandas as pd
        _old_to_datetime = pd.to_datetime

        def _patched_to_datetime(arg, *a, **kw):
            return _old_to_datetime(arg, format="ISO8601", utc=True)

        pd.to_datetime = _patched_to_datetime
        whole_dataset = wilds.get_dataset(
            dataset="fmow",
            root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/",
            download=True,
        )

        def get_groups(testset):
            groups = whole_dataset._eval_groupers['region'].metadata_to_group(testset.metadata_array)
            group_testset = []
            for i in range(5):
                idx = np.where(groups == i)[0]
                group_testset.append(WILDSSubset(testset, idx, None))

            return group_testset
        if time == 'id':
            dataset = whole_dataset.get_subset('id_val', transform=transform)
            intervention_dataset = whole_dataset.get_subset('id_val', transform=transform)
        elif time == 'time1':
            testsetv2 = whole_dataset.get_subset('val', transform=transform)
            region = int(dataset_name.split('-')[3].replace('region', ''))

            testsetv2_groups = get_groups(testsetv2)
            dataset = testsetv2_groups[region]
            intervention_dataset = testsetv2_groups[region]
        else:
            testsetv3 = whole_dataset.get_subset('test', transform=transform)
            region = int(dataset_name.split('-')[3].replace('region', ''))
            testsetv3_groups = get_groups(testsetv3)

            dataset = testsetv3_groups[region]
            intervention_dataset = testsetv3_groups[region]
    elif dataset_name == 'camelyon17-mean-hospital2':

        whole_dataset = wilds.get_dataset(
            "camelyon17", unlabeled=False, root_dir="/home/yxpengcs/PycharmProjects/DomainBed/domainbed/data/", split_scheme="official"
        )
        dataset = whole_dataset.get_subset("test", transform=transform)
        torch.manual_seed(0)
        indices = torch.randperm(len(dataset))[:4000]
        intervention_dataset = whole_dataset.get_subset("test", transform=transform)
        dataset = Subset(dataset, indices)
        intervention_dataset = Subset(intervention_dataset, indices)
    elif dataset_name == 'terra-incognita-38-mean' or dataset_name == 'terra-incognita-43-mean-location38' or dataset_name == 'terra-incognita-46-mean-location38' or dataset_name == 'terra-incognita-100-mean-location38':

        dataset = TerraIncognita('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val', "location_38", transform=transform)
        intervention_dataset = TerraIncognita('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val', "location_38", transform=transform)
    elif dataset_name == 'terra-incognita-38-mean-location43' or dataset_name == 'terra-incognita-43-mean' or dataset_name == 'terra-incognita-46-mean-location43' or dataset_name == 'terra-incognita-100-mean-location43':

        dataset = TerraIncognita('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val', "location_43", transform=transform)
        intervention_dataset = TerraIncognita('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val', "location_43", transform=transform)
        # local_rng = np.random.RandomState(seed=42)  # fixed seed here
        # subset_size = int(len(dataset) * 0.2)
        # indices = local_rng.choice(len(dataset), size=subset_size, replace=False)
        # intervention_dataset = TerraIncognita('/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', None, "location_43", transform=transform)
        # dataset = Subset(dataset, indices)
        # intervention_dataset = Subset(intervention_dataset, indices)
    elif dataset_name == 'terra-incognita-38-mean-location46' or dataset_name == 'terra-incognita-43-mean-location46'  or dataset_name == 'terra-incognita-46-mean' or dataset_name == 'terra-incognita-100-mean-location46':

        dataset = TerraIncognita(
            '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val',
            "location_46", transform=transform)
        intervention_dataset = TerraIncognita(
            '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val',
            "location_46", transform=transform)
        # local_rng = np.random.RandomState(seed=42)  # fixed seed here
        # subset_size = int(len(dataset) * 0.2)
        # indices = local_rng.choice(len(dataset), size=subset_size, replace=False)
        # intervention_dataset = TerraIncognita(
        #     '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', None,
        #     "location_46", transform=transform)
        # dataset = Subset(dataset, indices)
        # intervention_dataset = Subset(intervention_dataset, indices)
    elif dataset_name == 'terra-incognita-38-mean-location100' or dataset_name == 'terra-incognita-43-mean-location100' or dataset_name == 'terra-incognita-46-mean-location100' or dataset_name == 'terra-incognita-100-mean':

        dataset = TerraIncognita(
            '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val',
            "location_100", transform=transform)
        intervention_dataset = TerraIncognita(
            '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', 'val',
            "location_100", transform=transform)
        # local_rng = np.random.RandomState(seed=42)  # fixed seed here
        # subset_size = int(len(dataset) * 0.2)
        # indices = local_rng.choice(len(dataset), size=subset_size, replace=False)
        # intervention_dataset = TerraIncognita(
        #     '/home/yxpengcs/PycharmProjects/vit-spurious-robustness/domainbed/data/terra_incognita', None,
        #     "location_100", transform=transform)
        # dataset = Subset(dataset, indices)
        # intervention_dataset = Subset(intervention_dataset, indices)
    elif 'metashift-control' in dataset_name:

        ratio = dataset_name.split('-')[-1]
        dataset = get_metashift_dataset(split="test", root_dir=f'/home/yxpengcs/PycharmProjects/Moon-Shape-ICML-2023/experiments/metashift-{ratio}/data', transform=transform)
        intervention_dataset = get_metashift_dataset(split="test", root_dir=f'/home/yxpengcs/PycharmProjects/Moon-Shape-ICML-2023/experiments/metashift-{ratio}/data', transform=transform)
    elif dataset_name == 'metashift-mean':

        dataset = get_metashift_dataset(split="val", transform=transform)
        intervention_dataset = get_metashift_dataset(split="val", transform=transform)
    elif dataset_name == 'metashift-mean-ood':

        dataset = get_metashift_dataset(split="test", transform=transform)
        intervention_dataset = get_metashift_dataset(split="test", transform=transform)
    elif dataset_name == 'colored-object':
        if 'scratch' in model_name:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        elif 'IN-21k' in model_name:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        train1 = ColoredObject(root_dir='/home/yxpengcs/PycharmProjects/vMIB-circuit/datasets/ColoredObject', processor=processor, env='env_0')
        train2 = ColoredObject(root_dir='/home/yxpengcs/PycharmProjects/vMIB-circuit/datasets/ColoredObject', processor=processor, env='env_1')
        train1.combine(train2)
        dataset = train1
    elif dataset_name == 'colored-object-bg':
        if 'scratch' in model_name:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        elif 'IN-21k' in model_name:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        train1 = ColoredObject(root_dir='/home/yxpengcs/PycharmProjects/vMIB-circuit/datasets/ColoredObject', mode='bg', processor=processor, env='env_0')
        train2 = ColoredObject(root_dir='/home/yxpengcs/PycharmProjects/vMIB-circuit/datasets/ColoredObject', mode='bg', processor=processor, env='env_1')
        train1.combine(train2)
        dataset = train1
    elif dataset_name == 'waterbirds-mean':

        dataset = WaterbirdDataset(data_correlation=0.95, split="train",
                                   root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                   transform=transform, subset_fraction=0.2)
        intervention_dataset = WaterbirdDataset(data_correlation=0.95, split="train",
                                       root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                       transform=transform, subset_fraction=0.2)
    elif dataset_name == 'waterbirds-mean-ood':

        dataset = WaterbirdDataset(data_correlation=0.95, split="test",
                                   root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                   transform=transform, subset_fraction=0.2)
        intervention_dataset = WaterbirdDataset(data_correlation=0.95, split="test",
                                       root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                       transform=transform, subset_fraction=0.2)
    elif dataset_name == 'waterbirds-wg':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = WaterbirdDataset(data_correlation=0.95, split=split,
                                   root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                   transform=transform)
        ood_dataset = WaterbirdDataset(data_correlation=0.95, split=split,
                                       root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                       transform=transform)
        dataset.get_group(3)
        ood_dataset.get_group(2)
    elif dataset_name == 'waterbirds-bg':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ConsistencyDataset(transform,
            root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_bg',
            split=split)
    elif dataset_name == 'waterbirds-bg-water-group':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ConsistencyDataset(transform,
                                     root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_bg',
                                     split=split, label_select=1)
    elif dataset_name == 'waterbirds-bg-worst-group':

        dataset = ConsistencyDataset(transform,
                                     root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_bg',
                                     split=split, worst_group=True)
        intervention_dataset = ConsistencyDataset(transform,
                                                  root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_bg',
                                                  split=split, label_select=0)
    elif dataset_name == 'waterbirds-mean-worst-group':

        dataset = WaterbirdDataset(data_correlation=0.95, split=split,
                                       root_dir="/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets",
                                       transform=transform,  worst_group=True)
        intervention_dataset = ConsistencyDataset(transform,
                                     root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_bg',
                                     split=split, label_select=0)
    elif dataset_name == 'waterbirds-fg-water-group':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ConsistencyDataset(transform,
                                     root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_fg',
                                     split=split, label_select=1)
    elif dataset_name == 'waterbirds-fg-worst-group':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ConsistencyDataset(transform,
                                     root_dir='/home/yxpengcs/PycharmProjects/vit-spurious-robustness-modified/datasets/waterbird_fg',
                                     split=split, worst_group=True)
    elif dataset_name == 'colored-mnist':
        if split == 'train':
            split = 'all_train'
        dataset = ColoredMNIST(root='/home/yxpengcs/PycharmProjects/vision-grokking/new_data',
                               env=split, select_class='all',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                               ]))
        intervention_dataset = ColoredMNIST(
            root='/home/yxpengcs/PycharmProjects/vision-grokking/new_data',
            env='test', select_class='all',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
            ]))
    elif dataset_name in ('PACS-full-photo', 'PACS-full-sketch', 'PACS-full-cartoon', 'PACS-full-art_painting'):
        _pacs_base = '/home/yxpengcs/PycharmProjects/assaying-ood/webdata'
        _source = dataset_name.split('PACS-full-')[1]
        _json = f'{_pacs_base}/{_source}/split.json'
        dataset = PACSDataset(_json, 'all', transform=transform)
        intervention_dataset = PACSDataset(_json, 'all', transform=transform)
    else:
        ood_dataset = None
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    raw_datasets = {'train': train_dataset, 'validation': eval_dataset}
    return VisionEAPDataset(dataset, task=dataset_name, num_examples=num_examples, model_name=model_name, device=device, fragment=fragment), VisionEAPDataset(intervention_dataset, task=dataset_name, num_examples=num_examples, model_name=model_name, device=device, fragment=fragment)
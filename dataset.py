from typing import Optional
import random
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import PreTrainedTokenizer

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels


class EAPDataset(Dataset):
    def __init__(self, filepath:str, task:str='greater-than'):
        self.task = task
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.task == 'greater-than':
            return row['clean'], row['corrupted'], row['correct_idx']
        elif self.task == 'ioi':
            return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
        elif self.task == 'ewok':
            return row['Context1'], row['Context2'], [row['Target1'], row['Target2']]
        else:
            raise ValueError(f'Got invalid task: {self.task}')
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

background_colors = [(0, 100, 0), (188, 143, 143), (255, 0, 0), (255, 215, 0), (0, 255, 0), (65, 105, 225), (0, 225, 225), (0, 0, 255), (255, 20, 147), (160, 160, 160)]


class VisionEAPDataset(Dataset):
    task: str 
    tokenizer: PreTrainedTokenizer
    control: bool
    model_name: Optional[str]
    dataset: Dataset

    def __init__(self, dataset, num_examples:Optional[int]=None, task='waterbird',
                 control:Optional[bool]=False, counterfactual_type:Optional[str]=None, model_name: Optional[str] = None, device=None, fragment=None):
        self.dataset = dataset
        self.device = device
        self.control = control
        self.model_name = model_name
        self.task = task
        self.counterfactual_type = counterfactual_type
        
        # self.dataset = self.filter_dataset()
        #self.dataset = self.shuffle()
        if num_examples is not None:
            self.head(num_examples)
        if fragment is not None:
            self.get_fragment(fragment)
        
        # for when `control is True`
        self.answer_map = {}
        self.seed_offset = 0


    def __len__(self):
        return len(self.dataset)

    def get_fragment(self, n: int, total_fragments: int = 8):
        dataset_len = len(self.dataset)
        if not (0 <= n < total_fragments):
            raise ValueError(f"Fragment index {n} is out of bounds (should be between 0 and {total_fragments - 1})")

        fragment_size = dataset_len // total_fragments

        start = n * fragment_size
        if n == total_fragments - 1:
            end = dataset_len
        else:
            end = start + fragment_size

        self.dataset = [self.dataset[i] for i in range(start, end)]


    def shuffle(self):
        return self.dataset.shuffle()
    
    def head(self, n: int):
        if n <= len(self.dataset):
            self.dataset = [self.dataset[i] for i in range(n)]
        else:
            print("Warning: `num_examples` is greater than the size of the dataset! Returning the full dataset.")
            return self.dataset
        # if n <= len(self.dataset):
        #     self.dataset = self.dataset.select(range(n))
        # else:
        #     print("Warning: `num_examples` is greater than the size of the dataset! Returning the full dataset.")
        #     return self.dataset
    
    def tail(self, n: int):
        return [self.dataset[i] for i in range(len(self.dataset)-n, len(self.dataset))]
        

    def filter_dataset(self):
        if self.task == 'ioi':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) == 
                          len(self.tokenizer(f" {x['metadata']['subject']}", add_special_tokens=False).input_ids) and
                          len(self.tokenizer(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(f" {x['metadata']['random_c']}", add_special_tokens=False).input_ids)
            )
        elif self.task == 'mcqa':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(str(x[self.counterfactual_type]["choices"]["label"][x[self.counterfactual_type]["answerKey"]]),
                                             add_special_tokens=False).input_ids)
            )
        elif self.task == 'ewok':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["Target1"], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(x["Target2"], add_special_tokens=False).input_ids) and
                          x["Domain"] == self.example_domain
            )
        elif self.task.startswith('arithmetic'):
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(str(x["label"]), add_special_tokens=False).input_ids) == 1 and
                          x["random_counterfactual"] is not None and
                          x["random_counterfactual"]["prompt"] is not None and x["operator"] == self.example_domain and
                          len(self.tokenizer(str(x["random_counterfactual"]["label"]), add_special_tokens=False).input_ids) == 1
            )
        elif self.task == 'greater-than':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["clean"], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(x["corrupted"], add_special_tokens=False).input_ids)
            )
        elif self.task.startswith('arc'):
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(str(x[self.counterfactual_type]["choices"]["label"][x[self.counterfactual_type]["answerKey"]]),
                                             add_special_tokens=False).input_ids)
            )
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

        return filtered_dataset
    
    def __getitem__(self, index):
        def _make_control_answer(answer_idx, offset=0):
            if offset != 0:
                self.seed_offset += offset
            random.seed(index + self.seed_offset)

            if answer_idx not in self.answer_map:
                random_token = random.randint(1000, self.tokenizer.vocab_size-1000)
                existing_random_answers = set(self.answer_map.values())
                # keep resampling until we obtain a unique answer. maintains bijectivity
                while random_token in existing_random_answers:
                    self.seed_offset += 1
                    random.seed(index + self.seed_offset)
                    random_token = random.randint(1000, self.tokenizer.vocab_size-1000)
                self.answer_map[answer_idx] = random_token
                
            new_answer_idx = self.answer_map[answer_idx]
            return new_answer_idx

        row = self.dataset[index]
        if 'waterbird' in self.task:
            if isinstance(row[0], list):
                water_image, land_image = row[0]
                correct_idx = row[1]
                incorrect_idx = 1 - correct_idx
                image = water_image if correct_idx else land_image
                ctft_image = land_image if correct_idx else water_image
                place = row[2]
                if self.control:
                    correct_idx = _make_control_answer(correct_idx)
                    incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            else:
                image = row[0]
                correct_idx = row[1]
                incorrect_idx = 1 - correct_idx
                ctft_image = None
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'metashift' in self.task:
            image, y, env = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            place = row[2]
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'PACS' in self.task:
            image, y = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'camelyon17' in self.task:
            image, y, _ = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'cifar10' in self.task:
            image, y = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'fmow' in self.task:
            image, y, _ = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'iwildcam' in self.task:
            image, y, _ = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'terra-incognita' in self.task:
            image, y = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'colored-object-bg' in self.task:
            image, ctft_image, target, bg_color, ctft_bg_color = row
            correct_idx = target
            incorrect_idx = background_colors.index(ctft_bg_color)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'colored-mnist' in self.task:
            image, target = row
            color = 'red' if target == 0 else 'green'
            image_color = 'red' if len(torch.unique(image[0])) > 1 else 'green'
            orig_image = image.clone()
            ctft_image = image.clone()
            if color == image_color:
                ctft_image[[0, 1]] = ctft_image[[1, 0]]
            else:
                orig_image[[0, 1]] = orig_image[[1, 0]]
            correct_idx = target
            incorrect_idx = 1 - target
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return orig_image, ctft_image, [correct_idx, incorrect_idx]
        elif 'IN-set2' in self.task:
            image, y = row
            correct_idx = y
            incorrect_idx = 1 - y
            ctft_image = image
            if self.control:
                correct_idx = _make_control_answer(correct_idx)
                incorrect_idx = _make_control_answer(incorrect_idx, offset=1)
            return image, ctft_image, [correct_idx, incorrect_idx]
        elif 'IN-' in self.task:
            image, target = row
            correct_idx = target
            incorrect_idx = random.choice(self.dataset.ctft_class_ranges)
            ctft_image_path = self.dataset.ctft_dataset.samples[random.choice(self.dataset.ctft_class_index[incorrect_idx])][0]
            ctft_image = Image.open(ctft_image_path).convert("RGB")
            if self.dataset.processor is not None:
                ctft_image = self.dataset.processor(images=ctft_image, return_tensors="pt")["pixel_values"].squeeze(0) # Use processor
            if self.dataset.transform is not None:
                ctft_image = self.dataset.transform(ctft_image)
            return image, ctft_image, [correct_idx, incorrect_idx]
        else:
            raise NotImplementedError

    def to_dataloader(self, batch_size: int, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_EAP)

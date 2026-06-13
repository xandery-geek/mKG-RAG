import os
import csv
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.utils import resize_short_side


class EncyclopdicVQA(Dataset):
    _INAT21_MAPPING_FILE = "inaturalist_id2name.json"
    _IMAGE_MATCHING_MODE = ["first", "random", "multiple", "expansion"]

    def __init__(self, data_dir, inat21_dir, gldv2_dir, split, transform=None, image_matching_mode="first"):
        assert split in ["train", "val", "test"], "Split should be 'train', 'val' or 'test'"

        self.data_dir = data_dir
        self.inat21_dir = inat21_dir
        self.gldv2_dir = gldv2_dir
        self.split = split
        self.image_matching_mode = image_matching_mode

        assert image_matching_mode in self._IMAGE_MATCHING_MODE, \
            f"Image matching mode should be one of {self._IMAGE_MATCHING_MODE}"
        
        print(f"Loading images with image matching mode: {image_matching_mode}")

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(resize_short_side)
            ])

        self.header, self.data = self._load_data(os.path.join(data_dir, split + '.csv'))
        self.inat21_map = json.load(open(os.path.join(data_dir, self._INAT21_MAPPING_FILE))) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        assert self.image_matching_mode != "multiple", \
            "Multiple image matching mode is not supported for __getitem__ method"
        
        item = self.data[idx]
        question = item[self.header.index("question")]
        answer = item[self.header.index("answer")]
        image_path = self._get_image_path(
            item[self.header.index("dataset_name")],
            item[self.header.index("dataset_image_ids")]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, question, answer, image_path
    
    def _load_data(self, data_file):
        with open(data_file, 'r') as f:
            data = list(csv.reader(f))
        header = data[0]

        # match each question-answer pair with each image
        new_data = []
        for item in data[1:]:
            image_ids = item[header.index("dataset_image_ids")].split("|")

            if self.image_matching_mode == "multiple":
                # one question-answer pair can match multiple images
                item[header.index("dataset_image_ids")] = image_ids
                new_data.append(item)
            else:
                if self.image_matching_mode == "first":
                    # one question-answer pair matches the first image
                    image_ids = [image_ids[0]]
                elif self.image_matching_mode == "random":
                    # one question-answer pair matches a random image
                    image_ids = [random.choice(image_ids)]
                elif self.image_matching_mode == "expansion":
                    # expansion mode, where each question-answer pair is expanded to match all images
                    image_ids = image_ids
                else:
                    raise ValueError("Invalid image matching mode")

                for image_id in image_ids:
                    new_item = item.copy()
                    new_item[header.index("dataset_image_ids")] = image_id
                    new_data.append(new_item)

        return header, new_data
        
    def _get_image_path(self, dataset_name, image_id):
        if dataset_name == "inaturalist":
            image_path = os.path.join(self.inat21_dir, self.inat21_map[image_id])
        elif dataset_name == "landmarks":
            image_path = os.path.join(
                self.gldv2_dir, image_id[0], image_id[1], image_id[2], image_id + ".jpg"
            )
        else:
            raise ValueError("Invalid dataset name")

        return image_path
    
    def get_image_path(self, idx):
        item = self.data[idx]
        dataset_name = item[self.header.index("dataset_name")]
        dataset_image_ids = item[self.header.index("dataset_image_ids")]

        if isinstance(dataset_image_ids, list):
            # For multiple image matching mode, return all image paths
            image_paths = [self._get_image_path(dataset_name, img_id) for img_id in dataset_image_ids]
            return image_paths
        else:
            # For single image matching mode, return the single image path
            image_path = self._get_image_path(dataset_name, dataset_image_ids)
            return image_path
    
    def get_wiki_url(self, idx):
        item = self.data[idx]
        wiki_url = item[self.header.index("wikipedia_url")]
        wiki_url = wiki_url.split("|")
        return wiki_url
    
    def get_question_id(self, idx):
        return str(idx)
    
    def get_question_type(self, idx):
        item = self.data[idx]
        question_type = item[self.header.index("question_type")]
        return question_type
    
    def get_evidence_id(self, idx):
        item = self.data[idx]
        evidence_section_id = item[self.header.index("evidence_section_id")]
        evidence_section_id = evidence_section_id.split("|")
        return evidence_section_id
import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.utils import resize_short_side


class InfoseekVQA(Dataset):
    _ANNO_FILE_MAP = {
        "train": "infoseek_train_filtered.csv",
        "test": "infoseek_test_filtered.csv"
    }

    MISSING_IMAGE_ID = ["oven_05001795"]

    def __init__(self, data_dir, image_dir, split, transform=None, pick_number=None):
        assert split in ["train", "test"], "Split should be 'train' or 'test'"

        self.data_dir = data_dir
        self.image_dir = image_dir
        self.split = split
        self.pick_number = pick_number

        if self.pick_number is not None:
            print(f"Picking first {self.pick_number} items from the dataset")

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(resize_short_side)
            ])

        self.header, self.data = self._load_data(os.path.join(data_dir, self._ANNO_FILE_MAP[split]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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

        if self.pick_number is not None:
            data = data[1:self.pick_number + 1]
        else:
            data = data[1:]

        new_data = []
        for item in data:
            image_ids = item[header.index("dataset_image_ids")].split("|")
            for image_id in image_ids:
                if image_id in self.MISSING_IMAGE_ID:
                    continue
                new_item = item.copy()
                new_item[header.index("dataset_image_ids")] = image_id
                new_data.append(new_item)

        return header, new_data

    def _get_image_path(self, dataset_name, image_id):
        assert dataset_name in ["infoseek"], "Dataset name should be 'infoseek'"
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        if os.path.exists(image_path):
            return image_path
        
        image_path = os.path.join(self.image_dir, image_id + ".JPEG")
        if os.path.exists(image_path):
            return image_path
        
        raise FileNotFoundError(f"Image file not found for image_id: {image_id}")
    
    def get_image_path(self, idx):
        item = self.data[idx]
        image_path = self._get_image_path(
            item[self.header.index("dataset_name")],
            item[self.header.index("dataset_image_ids")]
        )
        return image_path
    
    def get_wiki_url(self, idx):
        item = self.data[idx]
        wiki_url = item[self.header.index("wikipedia_url")]
        wiki_url = wiki_url.split("|")
        return wiki_url
    
    def get_question_id(self, idx):
        item = self.data[idx]
        question_id = item[self.header.index("data_id")]
        return question_id

    def get_question_type(self, idx):
        item = self.data[idx]
        question_type = item[self.header.index("question_type")]
        return question_type
    
    def get_evidence_id(self, idx):
        return None
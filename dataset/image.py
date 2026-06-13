from PIL import Image
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, image_path
    
    def get_image_path(self, idx):
        return self.image_paths[idx]


class ImageTextDataset(Dataset):
    def __init__(self, images, texts, transform=None):
        assert len(images) == len(texts), "Image paths and texts must have the same length."
        
        self.images = images
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, text, None
    
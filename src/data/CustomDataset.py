import os
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_path in glob(os.path.join(class_path, '*.png')):
                images.append((img_path, self.class_to_index[class_name]))
        return images

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = read_image(img_path)  # Keep tensor format (C, H, W)
        image = image[:3]  # Keep only the first 3 channels, discarding alpha
        image = image.permute(1, 2, 0)  # Change to (H, W, C) for PIL conversion
        if self.transform:
            image = Image.fromarray(image.numpy())  # Convert to ndarray
            image = self.transform(image)
        return image, label
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        # Get the unique labels from the image filenames
        self.unique_labels = sorted(set(self.get_label(img_name) for img_name in self.images))
        
        # Create a dictionary to map labels to continuous integers
        self.label_map = {label: i for i, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.images)

    def get_label(self, img_name):
        try:
            # First method: Extract label after 'original_'
            start_label = img_name.find("original_") + len("original_")
            end_label = img_name[start_label:].find("_") + start_label
            label = int(img_name[start_label:end_label])
        except ValueError:
            # Second method: Extract label from the start of the filename to the first '_'
            label = int(img_name.split('_')[0])
        return label

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = read_image(img_name)
        
        label = self.get_label(self.images[idx])
        
        # Map the label to a continuous integer
        mapped_label = self.label_map[label]

        if self.transform:
            image = self.transform(image)
        
        return image, mapped_label
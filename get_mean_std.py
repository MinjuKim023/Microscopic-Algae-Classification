'''
This code is just for calculating mean and std of training set.

Calculated Mean:  tensor([0.6592, 0.6591, 0.6517])
Calculated Std:  tensor([0.0924, 0.0972, 0.1178])
'''

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_class import CustomImageDataset

def compute_mean_std(loader):
    '''
    Calculate mean and std from train data
    '''
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

# Temporarily load dataset
temp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ConvertImageDtype(torch.float32), 
])
temp_dataset = CustomImageDataset(root_dir='train_test_data/train', transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False)

mean, std = compute_mean_std(temp_loader)
print("Calculated Mean: ", mean)
print("Calculated Std: ", std)

normalize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean, std)
])

final_dataset = CustomImageDataset(root_dir='train_test_data/train', transform=normalize_transform)
final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)

for images, labels in final_loader:
    print(images.shape, labels)
    break  # Print first batch

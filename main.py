import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_class import CustomImageDataset
from utils import *
from model import *
import os

# def eval_model(model, data_loader, criterion, device):
#     # Evaluate the model on data from valloader
#     correct = 0
#     total = 0
#     val_loss = 0
#     model.eval()
#     with torch.no_grad():
#         for data in data_loader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).sum().item()
    
#     return val_loss / len(data_loader), 100 * correct / len(data_loader.dataset)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0

        num_classes = 12

        class_correct = dict.fromkeys(range(num_classes), 0)
        class_total = dict.fromkeys(range(num_classes), 0)
        TP = dict.fromkeys(range(num_classes), 0)
        FP = dict.fromkeys(range(num_classes), 0)
        TN = dict.fromkeys(range(num_classes), 0)
        FN = dict.fromkeys(range(num_classes), 0)

        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Update correct count and total count for accuracy
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Update TP, FP, TN, FN for each class
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                for class_index in range(num_classes):
                    if label == class_index:
                        if pred == class_index:
                            TP[class_index] += 1
                        else:
                            FN[class_index] += 1
                    else:
                        if pred == class_index:
                            FP[class_index] += 1
                        elif pred != label:
                            TN[class_index] += 1

        avg_loss = total_loss / len(data_loader)
        overall_accuracy = 100 * total_correct / total_samples

        # Calculate TPR and FPR for each class
        TPRs = {}
        FPRs = {}
        for class_index in range(num_classes):
            TPRs[class_index] = TP[class_index] / (TP[class_index] + FN[class_index]) if (TP[class_index] + FN[class_index]) > 0 else 0
            FPRs[class_index] = FP[class_index] / (FP[class_index] + TN[class_index]) if (FP[class_index] + TN[class_index]) > 0 else 0

        return avg_loss, overall_accuracy, TPRs, FPRs



def main(epochs = 100,
         model_class = 'CNN',
         batch_size = 128,
         learning_rate = 1e-4,
         l2_regularization = 0.0, val_split=0.2):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Transformations including normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                       # Resize images
        transforms.ConvertImageDtype(torch.float),         # Ensure float32 for image tensor
        transforms.Normalize([0.6592, 0.6591, 0.6517], [0.0924, 0.0972, 0.1178]),  # Normalize images
    ])

    train_dataset = CustomImageDataset(root_dir='train_test_data/train', transform=transform)
    test_dataset = CustomImageDataset(root_dir='train_test_data/test', transform=transform)
    unique_labels = torch.unique(torch.tensor([label for _, label in train_dataset]))

    # Create a dictionary to map labels to continuous integers
    label_map = {label.item(): i for i, label in enumerate(unique_labels)}

    # Map the labels in the train_dataset
    mapped_train_labels = [label_map[label] for _, label in train_dataset]
    train_dataset.targets = mapped_train_labels

    # Map the labels in the test_dataset
    mapped_test_labels = [label_map[label] for _, label in test_dataset]
    test_dataset.targets = mapped_test_labels
    
    total_samples = len(train_dataset)
    val_samples = int(total_samples * val_split)
    train_samples = total_samples - val_samples
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_samples, val_samples], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, the loss function and optimizer
    if model_class == 'CNN':
        net = CNN().to(device)
    elif model_class == 'ResNet': 
        net = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    elif model_class == 'UNet':
        net = UNet().to(device)
    elif model_class == 'MobileNet':
        net = MobileNet().to(device)
    
    print(net) # print model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)


    state_dict_name = f'weights/{os.getpid()}_{args.model_class}_weights.pth'
    stopper = EarlyStopping(mode='lower', patience=10, filename=state_dict_name, metric='loss') 

    # Train the model
    try:
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            net.train()
            for data in train_loader:
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                # print(outputs.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

       
            val_loss, val_acc, _, _ = eval_model(net, test_loader, criterion, device) 
            print('epoch - %d loss: %.3f accuracy: %.3f val_loss: %.3f val_acc: %.3f' % (epoch, running_loss / len(train_loader), 100 * correct / len(train_loader.dataset), val_loss, val_acc)) 
            
            early_stop = stopper.step(val_loss, net) 
            if early_stop:
                break 
            
        print('EarlyStopping! Finish training!') 
        print('Best epoch: {}'.format(epoch-stopper.counter)) 

        stopper.load_checkpoint(net) 
    except KeyboardInterrupt:
        pass
    
    net.eval()
    
    # Evaluate the model on the test set
    test_loss, test_acc, TPR, FPR = eval_model(net, test_loader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))
    print(f'TPR: {TPR} \nFPR: {FPR}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', 
                        type=int, 
                        default= 100,
                        help= "number of epochs the model needs to be trained for")
    parser.add_argument('--model_class', 
                        type=str, 
                        default= 'CNN', 
                        choices=['CNN', 'ResNet', 'UNet', 'MobileNet'],
                        help="specifies the model class that needs to be used for training, validation and testing.") 
    parser.add_argument('--batch_size', 
                        type=int, 
                        default= 128,
                        help = "batch size for training")
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default = 0.001,
                        help = "learning rate for training")
    parser.add_argument('--l2_regularization', 
                        type=float, 
                        default= 0.0,
                        help = "l2 regularization for training")
    
    args = parser.parse_args()
    main(**vars(args))
    # load_and_evaluate()

# python3 -u main.py >> log/$(date +"%y%m%d_%H%M")_CNN.txt 2>&1
# python3 -u main.py --model_class ResNet >> log/$(date +"%y%m%d_%H%M")_ResNet.txt 2>&1
# python3 -u main.py --model_class UNet >> log/$(date +"%y%m%d_%H%M")_UNet.txt 2>&1
# python3 -u main.py --model_class MobileNet >> log/$(date +"%y%m%d_%H%M")_MobileNet.txt 2>&1
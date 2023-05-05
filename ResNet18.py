import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set the directory paths for the train and test sets
train_dir = r'/home/tnchau/Tomato_Disease/train'
valid_dir = r'/home/tnchau/Tomato_Disease/valid'

train_image = glob(train_dir + '/*/*.JPG') + glob(train_dir + '/*/*.jpg') + glob(train_dir + '/*/*.jpeg') + glob(train_dir + '/*/*.JPEG')
valid_image = glob(valid_dir + '/*/*.JPG') + glob(valid_dir + '/*/*.jpg') + glob(valid_dir + '/*/*.jpeg') + glob(valid_dir + '/*/*.JPEG')

# Classes extraction
folders = glob(train_dir + '/*')

# Summary how many images for each class
import collections
train_labels = [os.path.basename(os.path.dirname(img_path)) for img_path in train_image]
train_class_counts = collections.Counter(train_labels)
print("Number of images per class in the training dataset:")
for class_name, count in train_class_counts.items():
    print("Class {}: {}".format(class_name, count))
    


# Define the data transformations
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create ImageFolder datasets for the train and validation sets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Create DataLoaders for the train and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
############


# Apply ResNet18 pretrain to train the model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the last layer to match the number of classes in your dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Calculate the average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print('Epoch {}/{} - Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))

    # Validation loop
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

        valid_loss = running_loss / len(valid_dataset)
        valid_acc = running_corrects.double() / len(valid_dataset)
        
    conf_matrix_list = conf_matrix.tolist()
    print('Validation - Loss: {:.4f} Acc: {:.4f}'.format(valid_loss, valid_acc))
    print('Confusion Matrix:')
    for row in conf_matrix_list:
        print(row)








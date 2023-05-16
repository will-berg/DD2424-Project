import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import pandas as pd

# Task 1
transform = transforms.Compose([
    transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
    transforms.ToTensor()
])
oxford_dataset = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True)#, transform=transform, )
oxford_dataset = torch.utils.data.Subset(oxford_dataset)

#i = 0
#for img, cl in oxford_dataset:
#  img.save(f"cat_explosion_{i}.png")
#  i += 1


#classes = oxford_dataset.classes
#
#dogs = ['American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'English Setter']
#cats = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
#
#dogs_labels = [oxford_dataset.class_to_idx[dog] for dog in dogs]
#cats_labels = [oxford_dataset.class_to_idx[cat] for cat in cats]
#
#breed_to_dog_cat = {}
#for dog_label in dogs_labels:
#    breed_to_dog_cat[dog_label] = 0
#
#for cat_label in cats_labels:
#    breed_to_dog_cat[cat_label] = 1

# Filter dataset for dogs and cats
dataloader = torch.utils.data.DataLoader(oxford_dataset, batch_size=4, shuffle=True)


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 37)


# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # Print the average loss for the epoch
    epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

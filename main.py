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
# oxford_dataset = torch.utils.data.Subset(oxford_dataset)

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
size = len(oxford_dataset)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 37)


# Training and testing
training_set, testing_set = torch.utils.data.random_split(oxford_dataset, [int(0.85 * size), int(0.15 * size)])

def train():
	model.train()

	learning_rate = 0.001
	n_epochs = 10

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	loss_function = nn.CrossEntropyLoss()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)

	for epoch in range(n_epochs):
		total_loss = 0
		for images, labels in training_loader:
			images = images.to(device)
			labels = labels.to(device)

			# Forward pass
			outputs = model(images)
			loss = loss_function(outputs, labels)

			# Backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		# Print the average loss for the epoch
		epoch_loss = total_loss / len(training_loader)
		print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

def test():
	model.eval()

	correct = 0
	testing_loader = torch.utils.data.DataLoader(testing_loader, batch_size=4, shuffle=True)

	with torch.no_grad():
		for images, labels in testing_loader:
			# Forward pass, returns model output
			probabilities = model(images)
			_, predictions = torch.max(probabilities, 1)
			correct += torch.sum(labels == predictions)

	accuracy = correct / len(testing_set)
	return accuracy


""" probabilities = torch.tensor([[0, 0.2, 0.3, 0.5], [0, 0.2, 0.3, 0.5]])
labels = torch.tensor([3, 2])

_, predictions = torch.max(probabilities, 1)
correct = torch.sum(labels == predictions)
print(predictions)

print(correct)
# same = torch.sum(list1 == list2)
# print(same / len(list1))
 """
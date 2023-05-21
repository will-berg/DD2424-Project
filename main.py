import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
import matplotlib.pyplot as plt


# Dataset directory
data_dir = './datasets/oxfordIIITPet/'

# Load the dataset for label extraction
oxford_dataset = datasets.OxfordIIITPet(root=data_dir, download=False, split="test")
# train_data = datasets.OxfordIIITPet(root=data_dir, download=False, split="trainval")
# add the datasets together
# full_dataset = torch.utils.data.ConcatDataset([oxford_dataset, train_data])


# Map new labels 'dog' and 'cat' to the original labels
dogs = ['American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'English Setter']
cats = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
dogs_labels = [oxford_dataset.class_to_idx[dog] for dog in dogs]
cats_labels = [oxford_dataset.class_to_idx[cat] for cat in cats]

label_map = {
    'dog': dogs_labels,
    'cat': cats_labels
}

# Define the target transform function
def target_transform(target):
    if target in label_map['dog']:
        return 0  # Assign label 0 for dogs
    elif target in label_map['cat']:
        return 1  # Assign label 1 for cats
    else:
        raise ValueError(f"Unknown label: {target}")

# Define the data transforms
transform = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.ToTensor()
	])

# Load the dataset with target transforms and change the classes
oxford_dataset_dog_cat = datasets.OxfordIIITPet(root=data_dir, transform=transform, target_transform=target_transform)
oxford_dataset_dog_cat.classes = ['dog', 'cat']
oxford_dataset_dog_cat_test = datasets.OxfordIIITPet(root=data_dir, transform=transform, target_transform=target_transform, split="test")
oxford_dataset_dog_cat_test.classes = ['dog', 'cat']
oxford_dataset_dog_cat = torch.utils.data.ConcatDataset([oxford_dataset_dog_cat, oxford_dataset_dog_cat_test])



# Parameters
learning_rate = 0.001
n_epochs = 25
batch_size = 16
loss_function = nn.CrossEntropyLoss()
model_name = f"{n_epochs}-{batch_size}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, training_set, validation_set, optimizer):
	model.train()
	model.to(device)

	training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)

	val_losses = []
	train_losses = []
	val_accs = []
	train_accs = []

	for epoch in range(n_epochs):
		train_total_loss = 0
		train_correct_predictions = 0
		train_number_of_predictions = 0
		for images, labels in training_loader:
			images = images.to(device)
			labels = labels.to(device)

			# Forward pass
			outputs = model(images)
			train_loss = loss_function(outputs, labels)

			# Backward pass and optimization
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

			train_total_loss += train_loss.item()

			_, predictions = torch.max(outputs, 1)
			train_correct_predictions += torch.sum(predictions == labels)
			train_number_of_predictions += len(labels)

		train_acc = train_correct_predictions / train_number_of_predictions
		train_accs.append(train_acc)
		# Print the average loss for the epoch
		train_epoch_loss = train_total_loss / len(training_loader)
		train_losses.append(train_epoch_loss)


		with torch.no_grad():
			val_epoch_loss = 0
			val_correct_predictions = 0
			val_number_of_predictions = 0
			for images, labels in validation_loader:
				images = images.to(device)
				labels = labels.to(device)

				# Forward pass, returns model output
				probabilities = model(images)
				val_loss = loss_function(probabilities, labels)
				val_epoch_loss += val_loss.item()

				_, predictions = torch.max(probabilities, 1)
				val_correct_predictions += torch.sum(predictions == labels)
				val_number_of_predictions += len(labels)

			val_acc = val_correct_predictions / val_number_of_predictions
			val_epoch_loss /= len(validation_loader)
			val_losses.append(val_epoch_loss)
			val_accs.append(val_acc)
		print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

	# Save the trained model
	torch.save(model.state_dict(), f"models/{model_name}.pth")

def test(model, testing_set):
	# Load the trained model
	model.load_state_dict(torch.load(f"models/{model_name}.pth"))
	model.to(device)
	model.eval()

	correct = 0
	testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

	with torch.no_grad():
		for images, labels in testing_loader:
			images = images.to(device)
			labels = labels.to(device)
			# Forward pass, returns model output
			probabilities = model(images)
			_, predictions = torch.max(probabilities, 1)
			correct += torch.sum(labels == predictions)

	acc = correct.item() / len(testing_set)
	print(f"Testing accuracy: {acc:.4f}")

def training_accuracy(model, training_set):
	# Load the trained model
	model.load_state_dict(torch.load(f"models/{model_name}.pth"))
	model.to(device)
	model.eval()

	correct = 0
	training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

	with torch.no_grad():
		for images, labels in training_loader:
			images = images.to(device)
			labels = labels.to(device)
			# Forward pass, returns model output
			probabilities = model(images)
			_, predictions = torch.max(probabilities, 1)
			correct += torch.sum(labels == predictions)

	acc = correct.item() / len(training_set)
	print(f"Training accuracy: {acc:.4f}")


def train_normal():
	transform = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.ToTensor()
	])
	oxford_dataset = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True, transform=transform)

	model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	model.fc = nn.Linear(512, 37)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# Training and testing
	train_set_size = int(0.90 * len(oxford_dataset))
	training_set, valtest_set = torch.utils.data.random_split(oxford_dataset, [train_set_size, len(oxford_dataset) - train_set_size])
	val_set_size = int(0.50 * len(valtest_set))
	validation_set, testing_set = torch.utils.data.random_split(valtest_set, [val_set_size, len(valtest_set) - val_set_size])
	train(model, training_set, validation_set, optimizer)
	test(model, testing_set)
	training_accuracy(model, training_set)

def train_normal_binary():
	oxford_dataset = oxford_dataset_dog_cat 

	model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	model.fc = nn.Linear(512, 2)

	for param in model.parameters():
		param.requires_grad = False
	for param in model.fc.parameters():
		param.requires_grad = True


	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# Training and testing
	train_set_size = int(0.90 * len(oxford_dataset))
	training_set, valtest_set = torch.utils.data.random_split(oxford_dataset, [train_set_size, len(oxford_dataset) - train_set_size])
	print(len(training_set))
	print(len(valtest_set))
	val_set_size = int(0.50 * len(valtest_set))
	validation_set, testing_set = torch.utils.data.random_split(valtest_set, [val_set_size, len(valtest_set) - val_set_size])
	print(len(validation_set))
	print(len(testing_set))

	train(model, training_set, validation_set, optimizer)
	test(model, testing_set)
	training_accuracy(model, training_set)





def find_nr_layers(freeze_BN_layers=True):
	transform = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.ToTensor()
	])
	oxford_dataset = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True, transform=transform)

	train_set_size = int(0.90 * len(oxford_dataset))
	training_set, valtest_set = torch.utils.data.random_split(oxford_dataset, [train_set_size, len(oxford_dataset) - train_set_size])
	val_set_size = int(0.50 * len(valtest_set))
	validation_set, testing_set = torch.utils.data.random_split(valtest_set, [val_set_size, len(valtest_set) - val_set_size])


	print("Last layer fine-tuned")
	model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	for param in model.parameters():
		param.requires_grad = False
	if not freeze_BN_layers:
		for module in model.modules():
			if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
				for param in module.parameters():
					param.requires_grad = True

	model.fc = nn.Linear(512, 37)
	model.fc.requires_grad = True
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	train(model, training_set, validation_set, optimizer)
	test(model, testing_set)

	print("Last 2 layers fine-tuned")
	model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	for param in model.parameters():
		param.requires_grad = False
	if not freeze_BN_layers:
		for module in model.modules():
			if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
				for param in module.parameters():
					param.requires_grad = True
	model.fc = nn.Linear(512, 37)
	model.fc.requires_grad = True
	model.layer4.requires_grad = True
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	train(model, training_set, validation_set, optimizer)
	test(model, testing_set)

	print("Last 3 layers fine-tuned")
	model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	for param in model.parameters():
		param.requires_grad = False
	if not freeze_BN_layers:
		for module in model.modules():
			if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
				for param in module.parameters():
					param.requires_grad = True
	model.fc = nn.Linear(512, 37)
	model.fc.requires_grad = True
	model.layer3.requires_grad = True
	model.layer4.requires_grad = True
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	train(model, training_set, validation_set, optimizer)
	test(model, testing_set)

	# Seems weird that layer 3 and 4 are third and second to last, but copilot chat insists that it is correct with motivation :shrug:

def find_learning_rate():
	transform = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.ToTensor()
	])
	oxford_dataset = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True, transform=transform)

	training_set, testing_set = torch.utils.data.random_split(oxford_dataset, [int(0.90 * len(oxford_dataset)), int(0.10 * len(oxford_dataset))])

	# Course search
	learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
	for learning_rate in learning_rates:
		print(f"Learning rate: {learning_rate}")
		model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
		model.fc = nn.Linear(512, 37)
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		train(model, training_set, optimizer)
		test(model, testing_set)

	# TODO: implement fine search based on results of above 

def find_data_augmentation():

	# Data augmentation 1
	transform1 = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.ToTensor()
	])

	# Data augmentation 2
	transform2 = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.RandomResizedCrop(300, scale=(0.5, 1.0)),
		transforms.ToTensor()
	])

	# Data augmentation 3
	transform3 = transforms.Compose([
		transforms.RandomCrop([300, 300], 1, pad_if_needed=True),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.RandomResizedCrop(300, scale=(0.5, 1.0)),
		transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
		transforms.ToTensor()
	])

	dataset1 = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True, transform=transform1)
	dataset2 = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True, transform=transform2)
	dataset3 = datasets.OxfordIIITPet("./datasets/oxfordIIITPet/", download=True, transform=transform3)

	training_set1, testing_set1 = torch.utils.data.random_split(dataset1, [int(0.90 * len(dataset1)), int(0.10 * len(dataset1))])
	training_set2, testing_set2 = torch.utils.data.random_split(dataset2, [int(0.90 * len(dataset2)), int(0.10 * len(dataset2))])
	training_set3, testing_set3 = torch.utils.data.random_split(dataset3, [int(0.90 * len(dataset3)), int(0.10 * len(dataset3))])

	print("Data augmentation 1")
	model1 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	model1.fc = nn.Linear(512, 37)
	optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
	train(model1, training_set1, optimizer1)
	test(model1, testing_set1)

	print("Data augmentation 2")
	model2 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	model2.fc = nn.Linear(512, 37)
	optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
	train(model2, training_set2, optimizer2)
	test(model2, testing_set2)

	print("Data augmentation 3")
	model3 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
	model3.fc = nn.Linear(512, 37)
	optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)
	train(model3, training_set3, optimizer3)
	test(model3, testing_set3)


def find_batch_norm():
	pass


def report_plots(): 


	plt.show()
	pass 


if __name__ == "__main__":
	#train_normal_binary()
	find_nr_layers(freeze_BN_layers=False)







# probabilities = torch.tensor([[0, 0.2, 0.3, 0.5], [0, 0.2, 0.3, 0.5]])
# labels = torch.tensor([3, 2])

# _, predictions = torch.max(probabilities, 1)
# correct = torch.sum(labels == predictions)

# print(correct.item())
# same = torch.sum(list1 == list2)
# print(same / len(list1))

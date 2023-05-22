import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import ConcatDataset, random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import time
import numpy as np
import kornia

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "./datasets/oxfordIIITPet"

transform = nn.Sequential(
    # random vertical flip
    kornia.augmentation.RandomVerticalFlip(p=0.4),
    # random rotation
    kornia.augmentation.RandomRotation(degrees=20),
    # random color jitter
    kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.3),
    # random grayscale
    kornia.augmentation.RandomGrayscale(p=0.2),
    # random blur
    kornia.augmentation.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.1),
    # random affine
    kornia.augmentation.RandomAffine((-15, 20), p=0.1),
    
  )

def load_dataset(binary_classification=False):
  transform = transforms.Compose([
    transforms.RandomCrop([400, 400], pad_if_needed=True, padding=1),
    transforms.ToTensor(),
  ])

  if binary_classification:
    oxford_dataset = datasets.OxfordIIITPet(root=data_dir, download=False, split="test")
    cats = ["Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"]
    dogs = ["American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Boxer", "Chihuahua", "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond", "Leonberger", "Miniature Pinscher", "Newfoundland", "Pomeranian", "Pug", "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"]

    dogs_labels = [oxford_dataset.class_to_idx[dog] for dog in dogs]
    cats_labels = [oxford_dataset.class_to_idx[cat] for cat in cats]

    label_map = {
        'dog': dogs_labels,
        'cat': cats_labels
    }
    # Define the target transform function for binary classification
    def target_transform(target):
      if target in label_map['dog']:
        return 0  # Assign label 0 for dogs
      elif target in label_map['cat']:
        return 1  # Assign label 1 for cats
      else:
        raise ValueError(f"Unknown label: {target}")

    train_dataset = datasets.OxfordIIITPet(root=data_dir, split="trainval", transform=transform, target_transform=target_transform)
    train_dataset.classes = ['dog', 'cat']
    test_dataset = datasets.OxfordIIITPet(root=data_dir, split="test", transform=transform, target_transform=target_transform)
    test_dataset.classes = ['dog', 'cat']
    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])


  else:
    train_dataset = datasets.OxfordIIITPet(root=data_dir, download=True, split="trainval", transform=transform)
    test_dataset = datasets.OxfordIIITPet(root=data_dir, download=True, split="test", transform=transform)
    dataset = ConcatDataset([train_dataset, test_dataset])

  # split the dataset
  training_data, validation_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])

  return training_data, validation_data, test_data

def freeze_pretrained_layers(model, n_layers_to_unfreeze, unfreeze_normalization=True):
  n = n_layers_to_unfreeze
  for param in model.parameters():
    param.requires_grad = False

  layers = []
  if n == 1:
    layers.append(model.fc)
  if n == 2:
    layers.append(model.layer4)
  if n == 3:
    layers.append(model.layer3)

  for layer in layers:
    for param in layer.parameters():
      param.requires_grad = True

  if unfreeze_normalization:
    for module in model.modules():
      if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        for param in module.parameters():
          param.requires_grad = True

def create_vit_model():
  model = models


def create_resnet_model(output_dimension=37, layers_to_fine_tune=1, fine_tune_normalization=True):
    # try model with more layers
    model = models.resnet34(weights="ResNet34_Weights.DEFAULT").to(device)
    # replace last layer
    model.fc = nn.Linear(512, output_dimension).to(device)
    # freeze the correct number of layers
    freeze_pretrained_layers(model, layers_to_fine_tune, fine_tune_normalization)

    return model

def evaluate(model, dataset, criterion):
  model.eval()

  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

  correct_predictions = 0
  running_loss = 0.0
  for inputs, labels in validation_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    running_loss += loss.item()

    _, predicted = torch.max(outputs, 1)
    correct_predictions += torch.sum(predicted == labels)


  validation_accuracy = correct_predictions/len(dataset)
  validation_loss = running_loss/len(validation_loader)
  return validation_accuracy, validation_loss

def train(model, training_data, validation_data, optimizer, scheduler, criterion, n_epochs=10):

  training_loader = torch.utils.data.DataLoader(training_data, batch_size=256, shuffle=True)

  training_metrics = []
  for epoch in range(n_epochs):
    time1 = time.time()
    model.train()
    correct_predictions = 0
    running_loss = 0.0
    for inputs, labels in training_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      inputs = transform(inputs)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      correct_predictions += torch.sum(predicted == labels)
    scheduler.step()

    training_accuracy = correct_predictions/len(training_data)
    training_loss = running_loss/len(training_loader)
    validation_accuracy, validation_loss = evaluate(model, validation_data, criterion)
    time2 = time.time()
    print(f"Epoch {epoch+1}/{n_epochs} | Training Loss: {training_loss:.4f} | Training Accuracy: {training_accuracy:.4f} | Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f} | In {time2-time1:.2f} seconds")
    training_metrics.append((training_loss, training_accuracy, validation_loss, validation_accuracy))

  torch.save(model.state_dict(), f"models/model.pth")

  return training_metrics

def plot_loss(training_loss, validation_loss, filename="loss.png"):
  plt.plot(training_loss, label="Training Loss")
  plt.plot(validation_loss, label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("plots/" + filename)

def plot_accuracy(training_accuracy, validation_accuracy, filename="accuracy.png"):
  # Convert tuple elements to floats
  training_accuracy = [float(i) for i in training_accuracy]
  validation_accuracy = [float(i) for i in validation_accuracy]

  plt.plot(training_accuracy, label="Training Accuracy")
  plt.plot(validation_accuracy, label="Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.savefig("plots/" + filename)

def plot(training_metrics, loss_filename="loss.png", accuracy_filename="accuracy.png"):
  training_loss, training_accuracy, validation_loss, validation_accuracy = zip(*training_metrics)
  plot_loss(training_loss, validation_loss, loss_filename)
  plt.close()
  plot_accuracy(training_accuracy, validation_accuracy, accuracy_filename)
  plt.close()

# Evaluate the most recently trained model on the test set
def test(model, test_data, criterion, load_from_pretrained=False):
  if load_from_pretrained:
    model.load_state_dict(torch.load("models/model.pth"))
  model.eval()
  test_accuracy, _ = evaluate(model, test_data, criterion)
  print(f"Test Accuracy: {test_accuracy:.4f}")



if __name__ == "__main__":
  do_binary_classification = False 
  load_from_pretrained = False 
  training_data, validation_data, test_data = load_dataset(binary_classification=do_binary_classification)

  model = create_resnet_model(
    output_dimension=2 if do_binary_classification else 37,
    layers_to_fine_tune=2,
    fine_tune_normalization=False
  )

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

  if not load_from_pretrained:
    training_metrics = train(
      model=model,
      training_data=training_data,
      validation_data=validation_data,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      n_epochs=50,
    )

    prefix = "binary_" if do_binary_classification else "multiclass_"
    plot(training_metrics, loss_filename=prefix + "loss.png", accuracy_filename=prefix + "accuracy.png")

  test(model, test_data, criterion, load_from_pretrained=load_from_pretrained)



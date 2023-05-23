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
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "./datasets/oxfordIIITPet"

def create_transform(random_vertical_flip=False, random_rotation=False, color_jitter=False, random_grayscale=False, random_gaussian_blue=False):
  transform = []
  if random_vertical_flip:
    transform.append(kornia.augmentation.RandomVerticalFlip(p=0.4))
  if random_rotation:
    transform.append(kornia.augmentation.RandomRotation(degrees=20))
  if color_jitter:
    transform.append(kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.3))
  if random_grayscale:
    transform.append(kornia.augmentation.RandomGrayscale(p=0.2))
  if random_gaussian_blue:
    transform.append(kornia.augmentation.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.1))

  return nn.Sequential(*transform)

def load_dataset(binary_classification=False):
  transform = transforms.Compose([
		transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    # transforms.RandomCrop([400, 400], pad_if_needed=True, padding=1),
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
    model = models.resnet152(weights="ResNet152_Weights.DEFAULT").to(device)
    # replace last layer
    model.fc = nn.Linear(2048, output_dimension).to(device)
    # freeze the correct number of layers
    freeze_pretrained_layers(model, layers_to_fine_tune, fine_tune_normalization)

    return model

def evaluate(model, dataset, criterion):
  model.eval()

  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

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

def train(model, training_data, validation_data, optimizer, scheduler, criterion, transform, n_epochs=10):

  training_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)

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
    training_metrics.append({
      "epoch": epoch+1,
      "training_loss": training_loss,
      "training_accuracy": float(training_accuracy),
      "validation_loss": validation_loss,
      "validation_accuracy": float(validation_accuracy),
    })

  return training_metrics


def save_training_metrics(training_metrics, filename="training_metrics.json"):
  path = "training_metrics/" + filename
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", ) as f:
    json.dump(training_metrics, f)

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
def test(model, test_data, criterion, load_from_pretrained=False, print_res=True):
  if load_from_pretrained:
    model.load_state_dict(torch.load("models/model.pth"))
  model.eval()
  test_accuracy, _ = evaluate(model, test_data, criterion)
  if print_res:
    print(f"Test Accuracy: {test_accuracy:.4f}")
  return float(test_accuracy)



if __name__ == "__main__":
  do_binary_classification = False
  training_data, validation_data, test_data = load_dataset(binary_classification=do_binary_classification)
  transform_properties = ["random_vertical_flip", "random_rotation", "color_jitter", "random_grayscale", "random_gaussian_blue"]
  finetune_layers = [1, 2, 3]
  freeze_normalization = [False, True]
  learning_rates = [1e-1, 1e-3, 1e-4]
  schedulers = [("exponential", lr_scheduler.ExponentialLR, {"gamma": 0.9}), ("cyclic", lr_scheduler.CyclicLR, {"base_lr": 1e-4, "max_lr": 1e-2, "cycle_momentum": False}), ("constant", "constant", {"lr": 1e-3})]

  transform = create_transform()
  def test_transform(transform_dict):
    transform = create_transform(**transform_dict)
    model = create_resnet_model(
      output_dimension=2 if do_binary_classification else 37,
      layers_to_fine_tune=1,
      fine_tune_normalization=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    training_metrics = train(
      model=model,
      training_data=training_data,
      validation_data=validation_data,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      transform=transform,
      n_epochs=15,
    )

    training_metrics = {
      "training_metrics": training_metrics
    }
    training_metrics["test_accuracy"] = test(model, test_data, criterion, print_res=False)

    filename = "no_transform" if len(transform_dict.keys()) == 0 else "_".join(transform_dict.keys())
    save_training_metrics(training_metrics, filename=f"transform_tests/{filename}.json")


  try:
    test_transform({})

    for transform_property in transform_properties:
      test_transform({transform_property: True})

    test_transform({transform_property: True for transform_property in transform_properties})
  except:
    pass

  def test_num_layers(fine_tune_layer, fine_tune_normalization=False):
    transform = create_transform(**{transform_property: True for transform_property in transform_properties})
    model = create_resnet_model(
      output_dimension=2 if do_binary_classification else 37,
      layers_to_fine_tune=fine_tune_layer,
      fine_tune_normalization=freeze_normalization
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    training_metrics = train(
      model=model,
      training_data=training_data,
      validation_data=validation_data,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      transform=transform,
      n_epochs=15,
    )

    training_metrics = {
      "training_metrics": training_metrics
    }
    training_metrics["test_accuracy"] = test(model, test_data, criterion, print_res=False)

    save_training_metrics(training_metrics, filename=f"num_layer_tests/{str(fine_tune_layer) + ('incl_normalization' if fine_tune_normalization else '')}.json")

  try:
    # test fine tune layers
    for fine_tune_normalization in [False, True]:
      for fine_tune_layer in finetune_layers:
        test_num_layers(fine_tune_layer, fine_tune_normalization=fine_tune_normalization)
  except:
    pass
  
  try:
    for scheduler_name, scheduler_class, scheduler_kwargs in schedulers:
      transform = create_transform(**{transform_property: True for transform_property in transform_properties})
      model = create_resnet_model(
        output_dimension=2 if do_binary_classification else 37,
        layers_to_fine_tune=1,
        fine_tune_normalization=False
      )

      criterion = nn.CrossEntropyLoss()
      if scheduler_class == "constant":
        optimizer = optim.Adam(model.parameters(), lr=scheduler_kwargs["lr"])
      else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)


      training_metrics = train(
        model=model,
        training_data=training_data,
        validation_data=validation_data,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        transform=transform,
        n_epochs=15,
      )

      training_metrics = {
        "training_metrics": training_metrics
      }
      training_metrics["test_accuracy"] = test(model, test_data, criterion, print_res=False)

      save_training_metrics(training_metrics, filename=f"scheduler_tests/{scheduler_name}.json")
  except:
    pass





    # plot(training_metrics, loss_filename=prefix + "loss.png", accuracy_filename=prefix + "accuracy.png")

# Final layer only, 10 epochs: 0.91
# Two last layers only, 10 epochs: 0.89
# Three last layers only, 10 epochs: 0.65

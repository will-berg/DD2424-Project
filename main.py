import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import ConcatDataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "./datasets/oxfordIIITPet"



def load_dataset(binary_classification=False):
  dogs = ['American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'English Setter']
  cats = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']

  transform = transforms.Compose([
    transforms.RandomCrop([400, 400], 1, pad_if_needed=True),
    transforms.ToTensor(),
  ])

  if not binary_classification:
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

def create_resnet_model(output_dimension=37, layers_to_fine_tune=1, fine_tune_normalization=True):
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT").to(device)
    # replace last layer
    model.fc = nn.Linear(512, output_dimension).to(device)
    # freeze the correct number of layers
    freeze_pretrained_layers(model, layers_to_fine_tune, fine_tune_normalization)

    return model
  
def evaluate(model, validation_data, criterion):
  model.eval()

  validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False)

  correct_predictions = 0
  running_loss = 0.0
  with torch.no_grad():
    for inputs, labels in validation_loader:
      inputs, labels = inputs.to(device), labels.to(device)

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      running_loss += loss.item()

      _, predicted = torch.max(outputs, 1)
      correct_predictions += torch.sum(predicted == labels)

    
    validation_accuracy = correct_predictions/len(validation_data)
    validation_loss = running_loss/len(validation_loader)
    return validation_accuracy, validation_loss

def train(model, training_data, validation_data, optimizer, criterion, n_epochs=10):

  training_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)

  training_metrics = []
  for epoch in range(n_epochs):
    model.train()
    correct_predictions = 0
    running_loss = 0.0
    for inputs, labels in training_loader:
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      correct_predictions += torch.sum(predicted == labels)
    
    training_accuracy = correct_predictions/len(training_data)
    training_loss = running_loss/len(training_loader)
    validation_accuracy, validation_loss = evaluate(model, validation_data, criterion)
    print(f"Epoch {epoch+1}/{n_epochs} | Training Loss: {training_loss:.4f} | Training Accuracy: {training_accuracy:.4f} | Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f}")
    training_metrics.append((training_loss, training_accuracy, validation_loss, validation_accuracy))

  return training_metrics



if __name__ == "__main__":
  training_data, validation_data, test_data = load_dataset(binary_classification=False)

  model = create_resnet_model(
    output_dimension=37, 
    layers_to_fine_tune=1, 
    fine_tune_normalization=False
  )

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  train(
    model=model,
    training_data=training_data,
    validation_data=validation_data,
    optimizer=optimizer, 
    criterion=criterion, 
    n_epochs=10
  )




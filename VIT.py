import torch
import torch.nn as nn
from datasets import load_dataset, load_metric, DatasetDict, Dataset
from torchvision import datasets, models, transforms
from torch.utils.data import ConcatDataset, random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import time
import numpy as np
import kornia
import json
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "./datasets/oxfordIIITPet"

def load_dataset(binary_classification=False):

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

    train_dataset = datasets.OxfordIIITPet(root=data_dir, split="trainval", target_transform=target_transform)
    train_dataset.classes = ['dog', 'cat']
    test_dataset = datasets.OxfordIIITPet(root=data_dir, split="test", target_transform=target_transform)
    test_dataset.classes = ['dog', 'cat']
    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
  else:
    train_dataset = datasets.OxfordIIITPet(root=data_dir, download=True, split="trainval")
    test_dataset = datasets.OxfordIIITPet(root=data_dir, download=True, split="test")
    dataset = ConcatDataset([train_dataset, test_dataset])

  # split the dataset
  training_data, validation_data, test_data = random_split(dataset, [0.6, 0.1, 0.3])

  return training_data, validation_data, test_data

training_data, validation_data, test_data = load_dataset() 

def convert_to_dict(sample):
  return {"image": sample[0], "label": sample[1]}

training_data = [convert_to_dict(sample) for sample in list(training_data)]
validation_data = [convert_to_dict(sample) for sample in list(validation_data)]
test_data = [convert_to_dict(sample) for sample in list(test_data)]

ds = DatasetDict({
  "train": Dataset.from_dict(training_data),
  "test": Dataset.from_dict(test_data),
  "validation": Dataset.from_dict(validation_data),
})

print(ds["train"][0])



model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)


def transform(example_batch):
  # example sample: (<PIL.Image.Image image mode=RGB size=500x333 at 0x1AE70063BD0>, 23)
  inputs = feature_extractor([x[0].convert("RGB") for x in example_batch], return_tensors="pt")
  inputs["labels"] = [x[1] for x in example_batch]
  return inputs

ds_p = ds.with_transform(transform)
print(ds_p["train"][0])
exit()

def collate_fn(batch):
  return {
      "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
      "labels": torch.tensor([example["labels"] for example in batch]),
  }

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

model = ViTForImageClassification.from_pretrained(
  model_name, 
  num_labels=len(labels),
  id2label={str(i): c for i, c in enumerate(labels)},
  label2id={c: i for i, c in enumerate(labels)},
)

print(model)

training_args = TrainingArguments(
  output_dir="./vit-base-oxford",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=1e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to="tensorboard",
  load_best_model_at_end=True,
)

from transformers import Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  data_collator=collate_fn,
  compute_metrics=compute_metrics,
  train_dataset=ds_p["train"],
  eval_dataset=ds_p["test"],
  tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate(ds_p["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


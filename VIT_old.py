import random
from PIL import ImageDraw, ImageFont, Image
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader
import numpy as np
import kornia
import torch.nn as nn
import json
import os



ds = load_dataset("pcuenq/oxford-pets")
ds = ds["train"].train_test_split(test_size=0.1, seed=42)

model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

all_labels = []
for example in ds["train"]:
    if example["label"] not in all_labels:

        all_labels.append(example["label"])


ds_labeltoid = {label: i for i, label in enumerate(all_labels)}
ds_idtolabel = {i: label for i, label in enumerate(all_labels)}

def transform(example_batch):
    inputs = feature_extractor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = [ds_labeltoid[y] for y in example_batch["label"]]
    return inputs

ds_p = ds.with_transform(transform)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
        "labels": torch.tensor([example["labels"] for example in batch]),
    }

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)



def train_model(transforms, learning_rate, scheduler_type, num_layers_frozen=-1):
  print("Training with transforms:", transforms, "learning_rate:", learning_rate, "scheduler_type:", scheduler_type)

  transform = []
  if "random_vertical_flip" in transforms:
    transform.append(kornia.augmentation.RandomVerticalFlip(p=0.4))
  if "random_rotation" in transforms:
    transform.append(kornia.augmentation.RandomRotation(degrees=20))
  if "color_jitter" in transforms:
    transform.append(kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.3))
  if "random_grayscale" in transforms:
    transform.append(kornia.augmentation.RandomGrayscale(p=0.2))
  if "random_gaussian_blur" in transforms:
    transform.append(kornia.augmentation.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.1))

  transform = nn.Sequential(*transform)

  model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=len(all_labels),
    id2label={str(i): c for i, c in enumerate(all_labels)},
    label2id={c: i for i, c in enumerate(all_labels)},
  )

  # if num_layers_frozen != 13:
  for param in model.parameters():
    param.requires_grad = False

  for param in model.classifier.parameters():
    param.requires_grad = True
  
  
  attention_layers = model.vit.encoder.layer
  if num_layers_frozen != -1:
    layers_to_unfreeze = 12 - num_layers_frozen
    for param in attention_layers[layers_to_unfreeze:].parameters():
      param.requires_grad = True

    

  
  if num_layers_frozen == 0:
    # unfreeze last normalization layer
    for param in model.vit.layernorm.parameters():
      param.requires_grad = True


  

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)

  training_args = TrainingArguments(
    output_dir="./vit-base-oxford",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=15,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=learning_rate,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
  )

  train_dataloader = DataLoader(ds_p["train"], batch_size=training_args.per_device_train_batch_size)
  eval_dataloader = DataLoader(ds_p["test"], batch_size=training_args.per_device_train_batch_size)

  #optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
  optimizer = torch.optim.Adam([
    {"params": model.vit.encoder.layer[:8].parameters(), "lr": 1e-4},
    {"params": model.vit.encoder.layer[8:].parameters(), "lr": 1e-3},
  ])



  #warmup_steps = int(0.1 * training_args.num_train_epochs * len(train_dataloader))
  #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch+1) / warmup_steps, 1.0))
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
  # linear scheduler
  if scheduler_type == "linear":
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / training_args.num_train_epochs)
  if scheduler_type == "exponential":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
  if scheduler_type == "constant":
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

  training_metrics = []

  for epoch in range(training_args.num_train_epochs):

    model.train()
    correct = 0
    total = 0
    tot_loss = 0
    for batch in train_dataloader:
      inputs = batch['pixel_values'].to(device)
      inputs = transform(inputs)
      labels = batch['labels'].to(device)

      optimizer.zero_grad()

      outputs = model(inputs, labels=labels)
      loss = outputs.loss
      loss.backward()

      tot_loss += loss.item()
      correct += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
      total += len(labels)

      optimizer.step()

    model.eval()

    with torch.no_grad():

      val_correct = 0
      val_total = 0
      val_loss = 0
      for batch in eval_dataloader:
        inputs = batch['pixel_values'].to(training_args.device)
        labels = batch['labels'].to(training_args.device)

        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()
        logits = outputs.logits
        val_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        val_total += len(labels)

    validation_accuracy, validation_loss, training_accuracy, training_loss = val_correct/val_total, val_loss/len(eval_dataloader), correct/total, tot_loss/len(train_dataloader)
    print(f"Epoch {epoch+1} | Training Loss: {training_loss:.4f} | Training Accuracy: {training_accuracy:.4f} | Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f}")
    training_metrics.append({
      "epoch": epoch+1,
      "training_loss": training_loss,
      "training_accuracy": training_accuracy,
      "validation_loss": validation_loss,
      "validation_accuracy": validation_accuracy
    })

    scheduler.step()

  def save_training_metrics(training_metrics, filename="training_metrics.json"):
    path = "training_metrics/" + filename
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", ) as f:
      json.dump(training_metrics, f)

  if num_layers_frozen == -1:
    save_training_metrics(training_metrics, "vit-tests/" + f"{'_'.join(transforms)}_{learning_rate}_{scheduler_type}.json")
  else:
    save_training_metrics(training_metrics, "vit-tests/frozen_layers/per_layer_lr/" + f"{num_layers_frozen}_{learning_rate}.json")

  return training_metrics[-1]["validation_accuracy"] 

#model = train_model([], 1e-4, "linear")
#print(list(model.vit.encoder.layer[5].attention.attention.query.parameters())[0][0,0])

all_transforms = ["random_vertical_flip", "random_rotation", "color_jitter", "random_grayscale", "random_gaussian_blur"]

best_acc = 0
best_lr = 1e-3

#for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
#  acc = train_model(all_transforms, lr, "linear")
#  if acc > best_acc:
#    best_acc = acc
#    best_lr = lr

#train_model([], best_lr, "linear")
#for transform_applied in ["random_vertical_flip", "random_rotation", "color_jitter", "random_grayscale", "random_gaussian_blur"]:
#  train_model([transform_applied], best_lr, "linear")
#
#train_model(all_transforms, best_lr, "linear")
#
#
#for scheduler_type in ["linear", "exponential", "constant"]:
#  train_model(all_transforms, best_lr, scheduler_type)
#
#for i in [2, 5, 8, 11]:
for i in [8, 11]:
  train_model(all_transforms, best_lr, "linear", i)

#train_model(all_transforms, best_lr, "linear", 12)





# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
#     train_dataset=ds_p["train"],
#     eval_dataset=ds_p["test"],
#     tokenizer=feature_extractor,
# )
# 
# train_results = trainer.train()
# trainer.save_model()
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()
# 
# 
# metrics = trainer.evaluate(ds_p["test"])
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)
# #
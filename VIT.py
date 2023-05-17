import random
from PIL import ImageDraw, ImageFont, Image
from datasets import load_dataset

ds = load_dataset("beans")

def show_example(ds, seed = 1234, examples_per_class = 3, size=(350, 350)):
    w, h = size
    labels = ds["train"].features["labels"].names
    grid = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("arial.ttf", size=20)

    for label_id, label in enumerate(labels):
        ds_slice = ds["train"].filter(lambda example: example["labels"] == label_id).shuffle(seed).select(range(examples_per_class))

        for i, example in enumerate(ds_slice):
            image = example["image"]
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255), font=font)

    return grid

from transformers import ViTFeatureExtractor

model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def transfrom(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs
  
prepared_ds = ds.with_transform(transfrom)

import torch

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
        "labels": torch.tensor([example["labels"] for example in batch]),
    }

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    return metric.compute(predictions=pred, references=labels)

from transformers import ViTForImageClassification

labels = ds["train"].features["labels"].names

model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: i for i, c in enumerate(labels)},
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./vit-base-beans",
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
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate(prepared_ds["validation"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

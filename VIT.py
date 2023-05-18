import random
from PIL import ImageDraw, ImageFont, Image
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
import torch
import numpy as np



ds = load_dataset("pcuenq/oxford-pets")
ds = ds["train"].train_test_split(test_size=0.1, seed=42)

model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

labels = []
for example in ds["train"]:
    if example["label"] not in labels:

        labels.append(example["label"])


ds_labeltoid = {label: i for i, label in enumerate(labels)}
ds_idtolabel = {i: label for i, label in enumerate(labels)}

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

model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: i for i, c in enumerate(labels)},
)

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


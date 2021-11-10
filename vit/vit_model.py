import os
import PIL
import shutil
import random
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

VIT_MODEL_NAME = "google/vit-base-patch16-224"
FOOD_MODEL_NAME = "nateraw/food"


# Select a model experiment
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=5)
parser.add_argument('--pretrain', help='Which pre-trained model to use (options: "original" (pre-trained model from the paper), "food101" (pre-trained model on food 101), default (no pre-training))', type=str)
parser.add_argument('--data_augmentation', help='Whether to run with data augmentation', dest='data_augmentation', action='store_true')
parser.set_defaults(data_augmentation=False)

args = parser.parse_args()

if args.pretrain == "original":
    MODEL_NAME = VIT_MODEL_NAME
    MODEL_DIR = "./original_model"
    
elif args.pretrain == "food101":
    MODEL_NAME = FOOD_MODEL_NAME
    MODEL_DIR = "./food_model"

else:
    MODEL_NAME = None
    MODEL_DIR = "./nopretrain_model"

if args.data_augmentation:
    MODEL_DIR += "_data_aug"


# Prepare dataset
train_dir = Path('../THFOOD50-v1/train/')
val_dir = Path('../THFOOD50-v1/val/')
test_dir = Path('../THFOOD50-v1/test/')

train_ds = ImageFolder(train_dir)
val_ds = ImageFolder(val_dir)
test_ds = ImageFolder(test_dir)


# Data Augmentation
if args.data_augmentation:
    train_rot90 = ImageFolder(train_dir, transform=lambda x: x.rotate(90))
    train_rot180 = ImageFolder(train_dir, transform=lambda x: x.rotate(180))
    train_rot270 = ImageFolder(train_dir, transform=lambda x: x.rotate(270))
    train_flip = ImageFolder(train_dir, transform=lambda x: x.transpose(PIL.Image.FLIP_LEFT_RIGHT))

    train_ds = train_ds + train_rot90 + train_rot180 + train_rot270 + train_flip


# Prepare label2id & id2label
label2id = {}
id2label = {}

for i, class_name in enumerate(test_ds.classes):
    label2id[class_name] = i
    id2label[i] = class_name

# Load pretrained model
if MODEL_NAME is not None:
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
else:
    config = ViTConfig.from_pretrained(
        VIT_MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL_NAME)
    model = ViTForImageClassification(config)

print("Number of parameters:", model.num_parameters())

# Preprocess images
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
 
    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors="pt")
        encodings["labels"] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings

collator = ImageClassificationCollator(feature_extractor)
train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collator, num_workers=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collator, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=8, collate_fn=collator, num_workers=2)


# Classifier module for training
class Classifier(pl.LightningModule):
    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters("lr", *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.top_1_acc = Accuracy()
        self.top_5_acc = Accuracy(top_k=5)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc_top_1 = self.top_1_acc(outputs.logits.argmax(1), batch["labels"])
        self.log(f"val_top_1_acc", acc_top_1, prog_bar=True)
        acc_top_5 = self.top_5_acc(outputs.logits.softmax(1), batch["labels"])
        self.log(f"val_top_5_acc", acc_top_5, prog_bar=True)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"test_loss", outputs.loss)
        acc_top_1 = self.top_1_acc(outputs.logits.argmax(1), batch["labels"])
        self.log(f"test_top_1_acc", acc_top_1, prog_bar=True)
        acc_top_5 = self.top_5_acc(outputs.logits.softmax(1), batch["labels"])
        self.log(f"test_top_5_acc", acc_top_5, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# Train the model
pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
trainer = pl.Trainer(
    gpus=1,
    precision=16,
    max_epochs=args.epochs,
    enable_checkpointing=True,
    default_root_dir=f"{MODEL_DIR}_trainer/",
    callbacks=[ModelCheckpoint(monitor="val_loss")],
)

start_time = time.time()
trainer.fit(classifier, train_loader, val_loader)
time_elapsed = time.time() - start_time
print("Time taken:", time_elapsed)

# Evaluate on validation set & test set
trainer.validate(classifier, val_loader, ckpt_path="best")
trainer.test(classifier, test_loader, ckpt_path="best")


# Save model
model.save_pretrained(MODEL_DIR)
feature_extractor.save_pretrained(MODEL_DIR)

# Copy logs from lightning_logs/ into ./model/runs/
tensorboard_logs_path = next(Path(trainer.logger.log_dir).glob("events.out*"))
model_logs_path = Path(MODEL_DIR) / "runs"
model_logs_path.mkdir(exist_ok=True, parents=True)
shutil.copy(tensorboard_logs_path, model_logs_path)


# Print out sample outputs
test_batch = next(iter(test_loader))
outputs = model(**test_batch)
print("Preds: ", outputs.logits.softmax(1).argmax(1))
print("Labels:", test_batch["labels"])


# Save sample images
plt.figure(figsize=(10, 10))

for i in range(9):
    class_idx = random.randint(0, len(test_ds.classes) - 1)
    class_name = model.config.id2label[class_idx]
    folder = test_ds.root / class_name
    image_path = random.choice(os.listdir(folder))
    image = Image.open(folder / image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(f"{class_name} ({predicted_class})")
    plt.axis("off")

plt.savefig(MODEL_DIR + "/sample_predictions.png")

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# =========================
# CONFIG
# =========================

DATASET_DIR = "data/labeled"

IMAGES_DIR = Path(DATASET_DIR) / "images"
LABELS_DIR = Path(DATASET_DIR) / "labels"

OUTPUT_DIR = "segformer_sidewalk_road_model"

MODEL_NAME = "nvidia/mit-b0"

NUM_CLASSES = 3
ID2LABEL = {
    0: "background",
    1: "sidewalk",
    2: "road",
}
LABEL2ID = {
    "background": 0,
    "sidewalk": 1,
    "road": 2,
}

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-5
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET
# =========================

class SidewalkRoadDataset(Dataset):
    def __init__(self, image_paths, label_paths, processor):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx])

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        mask = np.array(mask).astype(np.int64)

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collect_pairs():
    image_paths = sorted(list(IMAGES_DIR.glob("*.*")))

    pairs = []

    for image_path in image_paths:
        label_path = LABELS_DIR / f"{image_path.stem}.png"

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"Missing label for: {image_path.name}")

    if len(pairs) == 0:
        raise ValueError("No image/label pairs found.")

    return pairs


# =========================
# METRICS
# =========================

def compute_iou(preds, labels, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls

        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return ious


def evaluate(model, dataloader):
    model.eval()

    all_ious = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            preds = logits.argmax(dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for pred, label in zip(preds_np, labels_np):
                ious = compute_iou(pred, label, NUM_CLASSES)
                all_ious.append(ious)

    mean_ious = np.nanmean(np.array(all_ious), axis=0)
    mean_iou = np.nanmean(mean_ious)

    return mean_iou, mean_ious


# =========================
# TRAIN
# =========================

def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    pairs = collect_pairs()

    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_images, train_labels = zip(*train_pairs)
    val_images, val_labels = zip(*val_pairs)

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME,
        do_reduce_labels=False,
    )

    train_dataset = SidewalkRoadDataset(
        train_images,
        train_labels,
        processor,
    )

    val_dataset = SidewalkRoadDataset(
        val_images,
        val_labels,
        processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_miou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in loop:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        mean_iou, class_ious = evaluate(model, val_loader)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train loss: {avg_loss:.4f}")
        print(f"Val mIoU: {mean_iou:.4f}")

        for class_id, iou in enumerate(class_ious):
            print(f"{ID2LABEL[class_id]} IoU: {iou:.4f}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            print("Saving best model...")

            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)

    print("Training complete.")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
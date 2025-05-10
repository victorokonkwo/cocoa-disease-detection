import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip, Normalize,
    ColorJitter, ShiftScaleRotate, RandomBrightnessContrast
)
from src.config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS


class CocoaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None, is_train=True):
        """
        Args:
            csv_file (str): Path to the csv file with annotations
            img_dir (str): Directory with all the images
            transforms (callable, optional): Optional transform to be applied on a sample
            is_train (bool): Whether this is training set
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_train = is_train
        
        self.image_ids = self.df['Image_ID'].unique()
        
        if is_train:
            self.classes = self.df['class'].unique().tolist()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        annotations = self.df[self.df['Image_ID'] == img_id]
        
        boxes = []
        labels = []
        
        height, width = img.shape[:2]
        
        if self.is_train:
            for _, row in annotations.iterrows():
                x_min = row['xmin'] / width
                y_min = row['ymin'] / height
                x_max = row['xmax'] / width
                y_max = row['ymax'] / height
                
                x_min = max(0, min(1, x_min))
                y_min = max(0, min(1, y_min))
                x_max = max(0, min(1, x_max))
                y_max = max(0, min(1, y_max))
                
                # Skip invalid boxes
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.class_to_idx[row['class']])
        
        if self.transforms:
            if boxes:
                transformed = self.transforms(
                    image=img,
                    bboxes=boxes,
                    class_labels=labels
                )
                img = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["class_labels"]
            else:
                transformed = self.transforms(image=img)
                img = transformed["image"]
        
        target = {}
        if self.is_train:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
        
        return img, target if self.is_train else img_id


def get_transforms(is_train=True):
    """Get transforms for training or validation"""
    if is_train:
        return Compose([
            RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.3),
            ShiftScaleRotate(p=0.5, rotate_limit=15),
            RandomBrightnessContrast(p=0.5),
            ColorJitter(p=0.3),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})
    else:
        return Compose([
            RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.9, 1.0)),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})


def create_dataloaders(train_csv, test_csv, img_dir):
    """Create train and validation data loaders"""
    train_dataset = CocoaDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        transforms=get_transforms(is_train=True),
        is_train=True
    )
    
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    test_dataset = CocoaDataset(
        csv_file=test_csv,
        img_dir=img_dir,
        transforms=get_transforms(is_train=False),
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=test_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def collate_fn(batch):
    """Custom collate function for data loader"""
    images, targets = zip(*batch)
    return images, targets


def test_collate_fn(batch):
    """Custom collate function for test data loader"""
    images, img_ids = zip(*batch)
    return images, img_ids
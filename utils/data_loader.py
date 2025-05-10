import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.image_ids = self.annotations['image_id'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            image_annotations = self.annotations[self.annotations['image_id'] == image_id]

            boxes = []
            labels = []

            for _, row in image_annotations.iterrows():
                xmin, ymin = row['x_min'], row['y_min']
                xmax, ymax = row['x_max'], row['y_max']
                class_id = self.class_to_idx(row['class'])

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor(idx),
            }

            if self.transform:
                transformed = self.transform(image=image, bboxes=boxes, labels=labels)
                image = transformed['image']
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)

            return image, target
        else:
            if self.transform:
                image = self.transform(image=image)
            return image, image_id


def get_transform(train=True, img_size=640):
    if train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.RGBShift(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_data_loaders(train_csv, test_csv, img_dir, batch_size=8, img_size=640, num_workers=4):
    train_dataset = CocoaDataset(
        csv_file=train_csv, 
        img_dir=img_dir, 
        transform=get_transform(train=True, img_size=img_size), 
        train=True
    )
    
    test_dataset = CocoaDataset(
        csv_file=test_csv, 
        img_dir=img_dir, 
        transform=get_transform(train=False, img_size=img_size), 
        train=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))  # Custom collate function to handle variable-length targets
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        
        collate_fn=lambda x: tuple(zip(*x))  # Custom collate function to handle variable-length targets
    )
    
    return train_loader, test_loader
import torch
import torch.nn as nn
from ultralytics import YOLO
from src.config import MODEL_NAME, IMG_SIZE, DEVICE


class CocoaDiseaseDetector:
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize the YOLO model
        
        Args:
            num_classes (int): Number of classes to detect
            pretrained (bool): Whether to load pretrained weights
        """
        self.num_classes = num_classes
        self.device = DEVICE
        
        if pretrained:
            self.model = YOLO(f"{MODEL_NAME}.pt")
        else:
            self.model = YOLO(f"{MODEL_NAME}.yaml")
            
        self.model.overrides['task'] = 'detect'
        self.model.overrides['mode'] = 'train'
        self.model.overrides['batch'] = -1  # auto-batch
        self.model.overrides['imgsz'] = IMG_SIZE
        self.model.overrides['data'] = 'custom'
        self.model.overrides['epochs'] = 100
        self.model.overrides['patience'] = 25
        self.model.overrides['device'] = '0' if self.device == 'cuda' else 'cpu'
        self.model.overrides['workers'] = 8
        self.model.overrides['optimizer'] = 'AdamW'
        self.model.overrides['nc'] = num_classes  # number of classes
        
    def train(self, data_yaml, epochs=100, patience=25):
        """
        Train the model
        
        Args:
            data_yaml (str): Path to data.yaml file
            epochs (int): Number of epochs to train
            patience (int): Early stopping patience
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            patience=patience,
            batch=-1,
            imgsz=IMG_SIZE,
            pretrained=True,
            optimizer='AdamW',
            cos_lr=True
        )
    
    def predict(self, images, conf=0.25, iou=0.45):
        """
        Run inference on images
        
        Args:
            images: Images to run inference on
            conf (float): Confidence threshold
            iou (float): IoU threshold
        
        Returns:
            Predictions
        """
        return self.model.predict(
            images,
            conf=conf,
            iou=iou,
            imgsz=IMG_SIZE
        )
    
    def export(self, format='onnx', half=True):
        """
        Export the model to the specified format
        
        Args:
            format (str): Format to export to ('onnx', 'tflite', etc.)
            half (bool): Whether to use half precision
        
        Returns:
            Path to exported model
        """
        return self.model.export(format=format, half=half)
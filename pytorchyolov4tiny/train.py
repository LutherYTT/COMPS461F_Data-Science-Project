import os
import math
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import re
import time
import matplotlib.pyplot as plt
from pytorchyolov4tiny.models import *
from pytorchyolov4tiny.loss import *

import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '../..', 'config.yml')
config_path = os.path.normpath(config_path)
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# Hyperparameters
num_classes   = config['num_classes']
learning_rate = config['learning_rate']
num_epochs    = config['num_epochs']
batch_size    = config['batch_size']
accum_steps   = config['accum_steps']
img_size      = config['img_size']

train_list_file = "train.txt"
val_list_file   = "val.txt"
model_file = "../weights/baseline_model_epoch_30.pth"
#model_file = "/content/yolov4_tiny_weights_coco.pth"
cfg_file = "../config/yolov4-tiny.cfg"

# Array to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Create training and validation dataset
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
train_dataset = YoloDataset(train_list_file, img_size=img_size, transform=transform)
val_dataset   = YoloDataset(val_list_file, img_size=img_size, transform=transform)

# Setting the data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    #num_workers=8,
    pin_memory=True,
    collate_fn=custom_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    #num_workers=8,
    pin_memory=True,
    collate_fn=custom_collate_fn)

model = YOLOv4Tiny(cfg_file, num_classes=num_classes, img_size=img_size)

# Load the weights as the backbone
if os.path.exists(model_file):
    backbone_weights = torch.load(model_file, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in backbone_weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Backbone weights loaded successfully!")
else:
    print("No pre-trained backbone weights found. Training from scratch.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# freeze_layers(model, 0, 25)
# unfreeze_layers(model, 20, 25)

# Print requires_grad status to see which layers are freezed or unfreezed
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
# Using SGD with momentum (Experiment)
# optimizer = optim.SGD(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=learning_rate,
#     momentum=0.9,
#     weight_decay=5e-4  # L2 regularization
# )

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    optimizer.zero_grad()

    start = time.time()
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        predictions = model(images)
        loss, acc = yolo_loss(predictions, targets, model, device)
        loss = loss / accum_steps  # Scale loss for gradient accumulation
        loss.backward()
        running_loss += loss.item() * accum_steps
        running_acc += acc

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Step any remaining gradients
    if (i + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Compute average training metrics
    avg_train_loss = running_loss / len(train_loader)
    avg_train_acc = running_acc / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)

    epoch_time = time.time() - start
    print(f"Epoch時間分配 - 總耗時:{epoch_time:.2f}s")

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            for yolo_layer in model.yolo_layers:
                yolo_layer.train()

            predictions = model(images)
            loss, acc = yolo_loss(predictions, targets, model, device)
            val_loss += loss.item()
            val_acc += acc
            for yolo_layer in model.yolo_layers:
                yolo_layer.eval()
        for yolo_layer in model.yolo_layers:
            yolo_layer.eval()

    # Compute average validation metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_acc)

    # Display metrics
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    # Save the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint_path = f"./checkpoints/finetuned_yolov4_tiny_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Save the final model
final_path = "finetuned_yolov4_tiny_final.pth"
torch.save(model.state_dict(), final_path)
print(f"Final model saved at {final_path}")
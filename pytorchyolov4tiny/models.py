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

# Read image path, load the label
class YoloDataset(Dataset):
    def __init__(self, file_list, img_size=416, transform=None):
        self.img_paths = []
        # Read the image path from file (train.txt / text.txt)
        with open(file_list, 'r') as f:
            for line in f.read().splitlines():
                if line.strip():
                    self.img_paths.append(line.strip())
        self.img_size = img_size
        #　resize the image
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        image = self.transform(image)

        label_path = os.path.splitext(img_path)[0] + '.txt'
        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(x) for x in parts])
        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))

        # Scale boxes to resized image
        if boxes.size(0) > 0:
            scale_x = self.img_size / original_size[0]
            scale_y = self.img_size / original_size[1]
            boxes[:, 1] *= scale_x  # x_center
            boxes[:, 2] *= scale_y  # y_center
            boxes[:, 3] *= scale_x  # width
            boxes[:, 4] *= scale_y  # height
        return image, boxes

# Read the config file
def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        lines = f.readlines()

    # Read the layer configurations format
    layer_pattern = re.compile(r'\[(\w+)\]')
    config = {}
    module_defs = []
    module = {}

    for line in lines:
        if layer_pattern.match(line):
            if module:
                module_defs.append(module)
            module = {}
            module['type'] = layer_pattern.match(line).group(1)
        else:
            match = re.match(r'(\w+)\s*=\s*(.*)', line)
            if match:
                module[match.group(1)] = match.group(2)
    if module:
        module_defs.append(module)
    return module_defs

# According to config file to structure the layers
class YOLOv4Tiny(nn.Module):
    def __init__(self, cfg_file, num_classes, img_size=832):
        super(YOLOv4Tiny, self).__init__()
        self.num_classes = num_classes
        self.module_defs = parse_cfg(cfg_file)
        self.img_size = img_size
        if self.module_defs[0]['type'] == 'net':
            self.module_defs.pop(0)
        self.module_list = self.create_modules(self.module_defs)
        self.yolo_layers = [layer for layer in self.module_list if hasattr(layer, 'is_yolo')]


    def create_modules(self, module_defs):
        module_list = nn.ModuleList()
        in_channels = 3
        output_channels = [3]
        current_stride = 1
        strides = [current_stride]
        all_layer_index = 0

        for i, mdef in enumerate(module_defs):
            all_layer_index += 1
            if mdef['type'] == 'net':
                # Create a dummy module to maintain index alignment.
                dummy = nn.Identity()
                module_list.append(dummy)
                output_channels.append(in_channels)
                strides.append(current_stride)
                continue
            if mdef['type'] == 'convolutional':
                #print(f"Adding convolutional layer with in_channels={in_channels} and filters={mdef['filters']}")
                bn = int(mdef.get('batch_normalize', 0))
                filters = int(mdef['filters'])
                kernel_size = int(mdef['size'])
                stride = int(mdef['stride'])
                pad = int(mdef['pad'])
                activation = mdef['activation']
                padding = (kernel_size - 1) // 2 if pad else 0
                conv = nn.Conv2d(in_channels, filters, kernel_size, stride, padding, bias=not bn)
                modules = nn.Sequential()
                modules.add_module(f'conv_{i}', conv)
                if bn:
                    bn_layer = nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5)
                    modules.add_module(f'bn_{i}', bn_layer)
                if activation == 'leaky':
                    modules.add_module(f'leaky_{i}', nn.LeakyReLU(0.1))
                in_channels = filters
                output_channels.append(in_channels)
                # record the stride
                layer_stride = stride
                current_stride *= layer_stride
                strides.append(current_stride)
                module_list.append(modules)
                continue
            elif mdef['type'] == 'maxpool':
                kernel_size = int(mdef['size'])
                stride = int(mdef['stride'])
                modules = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                output_channels.append(in_channels)

                if kernel_size == 2 and stride == 1:
                    layer_stride = 1
                else:
                    layer_stride = stride
                    current_stride *= layer_stride
                strides.append(current_stride)
                module_list.append(modules)
                continue
            elif mdef['type'] == 'upsample':
                stride = int(mdef['stride'])
                modules = nn.Upsample(scale_factor=stride, mode='nearest')
                output_channels.append(in_channels)
                # record the stride
                layer_stride = stride
                current_stride *= layer_stride
                strides.append(current_stride)
                module_list.append(modules)
                continue
            elif mdef['type'] == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                layers = [l if l > 0 else i + l for l in layers]
                route_channels = [output_channels[l] for l in layers]
                #print(f"Creating route module at config index {i}: raw layers {mdef['layers']} -> converted layers {layers}, available output_channels: {[output_channels[l] for l in layers]}")
                for l in layers:
                    if l >= len(output_channels) or l < 0:
                        raise ValueError(f"Invalid route layer index {l} at layer {i}")
                route_channels = [output_channels[l] for l in layers]
                filters = sum(route_channels)
                if 'groups' in mdef:
                    groups = int(mdef['groups'])
                    group_id = int(mdef['group_id'])
                    assert all(c % groups == 0 for c in route_channels), "Channels must be divisible by groups"
                    grouped_channels = [c // groups for c in route_channels]
                    in_channels = sum(grouped_channels)
                else:
                    in_channels = sum(route_channels)
                # record the stride
                output_channels.append(in_channels)
                current_stride = max(strides[l] for l in layers)
                strides.append(current_stride)

                module = nn.Module()
                module.layers = layers
                module.groups = int(mdef.get('groups', 1))
                module.group_id = int(mdef.get('group_id', 0))
                module_list.append(module)
                continue
            elif mdef['type'] == 'yolo':
                mask = [int(x) for x in mdef['mask'].split(',')]
                anchors = [int(x) for x in mdef['anchors'].split(',')]
                anchors = [(anchors[i*2], anchors[i*2+1]) for i in range(len(anchors)//2)]
                scale_factor = self.img_size / 416  # 832 / 416 = 2
                anchors = [(a[0] * scale_factor, a[1] * scale_factor) for a in anchors]
                anchors = [anchors[i] for i in mask]
                modules = YOLOLayer(anchors, self.num_classes, is_training=True)
                output_channels.append(in_channels)
                modules.stride = current_stride
                strides.append(current_stride)
                module_list.append(modules)
                continue
            else:  # 'net' layer
                output_channels.append(in_channels)
                strides.append(current_stride)
                #module_list.append(modules)
                continue
        return module_list

    def forward(self, x):
        layer_outputs = []
        yolo_outputs = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            layer_type = mdef['type']

            if layer_type in ['convolutional', 'maxpool', 'upsample']:
                x = module(x)
                layer_outputs.append(x)
            elif layer_type == 'route':
                # Adjust negative indices if necessary
                layers = module.layers
                groups = module.groups
                group_id = module.group_id
                if groups > 1:
                    x_parts = []
                    for l in layers:
                        layer_out = layer_outputs[l]
                        n_channels = layer_out.size(1)
                        assert n_channels % groups == 0, f"Channels {n_channels} must be divisible by groups {groups} at route layer {i}"
                        n_channels_per_group = n_channels // groups
                        start = group_id * n_channels_per_group
                        end = start + n_channels_per_group
                        x_parts.append(layer_out[:, start:end])
                    x = torch.cat(x_parts, dim=1)
                else:
                    x = torch.cat([layer_outputs[l] for l in layers], dim=1)

                layer_outputs.append(x)
                #print(f"Route layer at index {i}, layers={layers}, groups={groups}, group_id={group_id}, output_channels={x.size(1)}")
            elif layer_type == 'yolo':
                x = module(x, self.img_size)
                yolo_outputs.append(x)
                layer_outputs.append(x)

        return yolo_outputs

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, is_training):
        super(YOLOLayer, self).__init__()
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
        self.new_coords = False
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.is_training = is_training
        self.ignore_thres = 0.5
        self.no = self.bbox_attrs
        self.stride = None

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.register_buffer('grid', torch.zeros(1))
        self.register_buffer('anchors', anchors)

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        stride = img_size // x.size(2)
        self.stride = stride

        if not self.training:  # Inference
            bs, _, ny, nx = x.shape
            x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.grid.shape[2:4] != (ny, nx):
                self.grid = self._make_grid(nx, ny).to(x.device)
            xy = (x[..., 0:2].sigmoid() + self.grid) * stride
            wh = (x[..., 2:4].sigmoid() * 2) ** 2 * self.anchor_grid
            y = torch.cat((xy, wh, x[..., 4:].sigmoid()), -1)
            # Debug statements
            #print(f"Anchor grid: {self.anchor_grid}")
            #print(f"Raw wh offsets: {x[..., 2:4].mean().item():.4f} {x[..., 2:4].max().item():.4f}")
            #print(f"Scaled wh: {wh.mean().item():.4f} {wh.max().item():.4f}")
            #print("Raw wh predictions:", x[..., 2:4].sigmoid().mean())
            #print("Anchors used:", self.anchors)
            return y.view(bs, -1, self.no)
        else:  # Training
            return x

    @property
    def is_yolo(self):
        return True
		
# Define a custom collate function
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, targets

def freeze_layers(model, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        module = model.module_list[i]
        for param in module.parameters():
            param.requires_grad = False

def unfreeze_layers(model, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        module = model.module_list[i]
        for param in module.parameters():
            param.requires_grad = True
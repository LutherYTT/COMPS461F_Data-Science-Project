import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os
from pytorchyolov4tiny.models import YOLOv4Tiny, YOLOLayer

def xywh_to_xyxy(boxes):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=1)

# Hyperparameters
num_classes = config['num_classes']
img_size = config['img_size']
cfg_file = config['cfg_file_path']
model_path = config['model_path']
conf_threshold = config['conf_threshold']
iou_threshold = config['iou_threshold']
top_conf = config['top_conf']

image_path = "./input.png"
output_txt_path = "/content/detections_output.txt"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = YOLOv4Tiny(cfg_file, num_classes=num_classes, img_size=img_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load and prepare the image
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Run the model
with torch.no_grad():
    # List of predictions from each YOLO layer
    predictions = model(image_tensor)  

# Concatenate predictions directly
# Shape: (total_detections, 5 + num_classes)
detections = torch.cat(predictions, dim=1)[0] 

# Process detections
boxes = detections[:, :4]  # [x, y, w, h] in pixel coordinates
objectness = detections[:, 4]  # Objectness scores
class_scores = detections[:, 5:]  # Class scores
max_class_score, class_id = torch.max(class_scores, dim=1)
conf = objectness * max_class_score

# Filter detections
mask = conf > conf_threshold
selected_boxes = boxes[mask]
selected_conf = conf[mask]
selected_class_id = class_id[mask]
selected_class_scores = class_scores[mask]
selected_objectness = objectness[mask]

# Debug confidence components
if selected_boxes.size(0) > 0:
    print("Debugging confidence components:")
    print(f"Objectness: {selected_objectness}")
    print(f"Max class scores: {max_class_score[mask]}")
    print(f"Confidences: {selected_conf}")

# Apply NMS per class
final_detections = []
for cls in range(num_classes):
    cls_mask = selected_class_id == cls
    if cls_mask.sum() == 0:
        continue
    cls_boxes = selected_boxes[cls_mask]
    cls_conf = selected_conf[cls_mask]
    cls_class_scores = selected_class_scores[cls_mask]
    cls_boxes_xyxy = xywh_to_xyxy(cls_boxes)
    keep = torchvision.ops.nms(cls_boxes_xyxy, cls_conf, iou_threshold)
    kept_boxes = cls_boxes[keep]
    kept_conf = cls_conf[keep]
    kept_class_id = torch.full((keep.size(0),), cls, dtype=torch.int64, device=device)
    kept_class_scores = cls_class_scores[keep]
    detections_cls = torch.cat(
        (kept_boxes, kept_conf[:, None], kept_class_id[:, None].float(), kept_class_scores), dim=1
    )
    final_detections.append(detections_cls)

# Combine and sort detections
if final_detections:
    final_detections = torch.cat(final_detections, dim=0)
    sorted_indices = torch.argsort(final_detections[:, 4], descending=True)
    final_detections = final_detections[sorted_indices][:top_conf]
else:
    final_detections = torch.zeros(0, 6 + num_classes, device=device)

# Save normalized detections
if final_detections.size(0) > 0:
    boxes_normalized = final_detections[:, :4] / img_size
    conf = final_detections[:, 4]
    class_id = final_detections[:, 5].long()
    class_scores = final_detections[:, 6:]
    with open(output_txt_path, 'w') as f:
        for i in range(final_detections.size(0)):
            line = [f"{class_id[i].item()}", f"{conf[i].item():.6f}"]
            line.extend([f"{class_scores[i, j].item():.6f}" for j in range(num_classes)])
            line.extend([f"{boxes_normalized[i, j].item():.6f}" for j in range(4)])
            f.write(" ".join(line) + "\n")

# Output detections
print(f"Detections for {image_path}:")
if final_detections.size(0) > 0:
    for i in range(final_detections.size(0)):
        x, y, w, h = final_detections[i, :4]
        conf = final_detections[i, 4]
        class_id = final_detections[i, 5].long()
        print(f"Class: {class_id.item()}, Confidence: {conf.item():.4f}, "
              f"Box: [x={x.item():.2f}, y={y.item():.2f}, w={w.item():.2f}, h={h.item():.2f}]")
else:
    print("No detections found.")
print(f"Detections saved to {output_txt_path}")
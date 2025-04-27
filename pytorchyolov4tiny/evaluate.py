import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import glob
import re
from pytorchyolov4tiny.models import *

# Calculate the IoU for both bounding boxes
def compute_iou(box1, box2):
    # box format: [x_center, y_center, width, height]
    # Convert to x1y1x2y2 format
    box1 = [
        box1[0] - box1[2]/2,
        box1[1] - box1[3]/2,
        box1[0] + box1[2]/2,
        box1[1] + box1[3]/2
    ]
    box2 = [
        box2[0] - box2[2]/2,
        box2[1] - box2[3]/2,
        box2[0] + box2[2]/2,
        box2[1] + box2[3]/2
    ]

    # Calculate the intersection region
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

# Processing model output as a bounding box list
def process_predictions(predictions, conf_threshold=0.5, iou_threshold=0.5, img_size=832):
    detections = torch.cat([pred for pred in predictions], dim=1)
    batch_size = detections.size(0)
    pred_boxes_list = []

    for i in range(batch_size):
        img_detections = detections[i]
        mask = img_detections[:, 4] >= conf_threshold
        img_detections = img_detections[mask]

        if img_detections.size(0) == 0:
            pred_boxes_list.append([])
            continue

        boxes = img_detections[:, :4]
        class_scores = img_detections[:, 5:]
        class_ids = torch.argmax(class_scores, dim=1)
        objectness = img_detections[:, 4]
        confidences = objectness * torch.max(class_scores, dim=1)[0]

        # Convert to xyxy format
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2

        # NMS
        keep_indices = torchvision.ops.nms(boxes_xyxy, confidences, iou_threshold)
        boxes = boxes[keep_indices]
        class_ids = class_ids[keep_indices]
        confidences = confidences[keep_indices]

        pred_boxes = []
        for j in range(boxes.size(0)):
            pred_boxes.append({
                'class_id': class_ids[j].item(),
                'confidence': confidences[j].item(),
                'bbox': boxes[j].cpu().numpy().tolist()
            })
        pred_boxes_list.append(pred_boxes)

    return pred_boxes_list

# Calculation of assessment indicators
def calculate_metrics(all_gts, all_preds, num_classes, iou_threshold=0.5):
    # Calculate mAP
    aps = []
    for cls in range(num_classes):
        cls_preds = [p for p in all_preds if p['class_id'] == cls]
        cls_gts = [g for g in all_gts if g['class_id'] == cls]

        # Sort by confidence
        cls_preds_sorted = sorted(cls_preds, key=lambda x: x['confidence'], reverse=True)
        tp = np.zeros(len(cls_preds_sorted))
        fp = np.zeros(len(cls_preds_sorted))
        gt_matched = [False] * len(cls_gts)

        for pred_idx, pred in enumerate(cls_preds_sorted):
            max_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(cls_gts):
                if not gt_matched[gt_idx]:
                    iou = compute_iou(pred['bbox'], gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = gt_idx

            if max_iou >= iou_threshold and best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        eps = 1e-6
        recall = tp_cumsum / (len(cls_gts) + eps)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + eps)

        # Calculate AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            if np.any(mask):
                ap += np.max(precision[mask]) / 11
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0
    mAP = float(mAP)  # Ensure conversion to scalar

    # Calculation of global targets
    selected_preds = [p for p in all_preds if p['confidence'] >= 0.5]
    gt_matched = [False] * len(all_gts)
    tp = 0
    fp = 0

    for pred in selected_preds:
        max_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(all_gts):
            if gt['class_id'] != pred['class_id']:
                continue

            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > max_iou and not gt_matched[gt_idx]:
                max_iou = iou
                best_gt_idx = gt_idx

        if max_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = len([g for g in all_gts if not gt_matched[all_gts.index(g)]])

    # Ensure that all indicators are converted to floating point numbers
    precision = float(tp / (tp + fp + 1e-6))
    recall = float(tp / (tp + fn + 1e-6))
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-6))

    return (float(mAP), float(precision), float(recall), float(f1))

# Evaluating individual models
def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    all_gts = []
    all_preds = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            predictions = model(images)
            pred_boxes_list = process_predictions(predictions, img_size=832)

            # Handling grund truth labels
            for i in range(len(targets)):
                gt_boxes = targets[i].cpu().numpy()
                for box in gt_boxes:
                    class_id = int(box[0])
                    x_center = box[1] * 832
                    y_center = box[2] * 832
                    w = box[3] * 832
                    h = box[4] * 832
                    all_gts.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, w, h]
                    })

            # Handling of predicted results
            for boxes in pred_boxes_list:
                for box in boxes:
                    all_preds.append(box)

    return calculate_metrics(all_gts, all_preds, num_classes)

# Hyperparameter
num_classes = config['num_classes']
img_size = config['img_size']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg_file = config['cfg_file_path']


test_list_file = "test.txt"

# Load test dataset
test_dataset = YoloDataset(test_list_file, img_size=img_size)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn
)

# Get all model checkpoints
# model_files = sorted(
#     glob.glob("/content/baseline_checkpoint/baseline_3_Checkpoint/finetuned_yolov4_tiny_epoch_*.pth"),
#     key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1))
# )
# model_files = ["/content/baseline_checkpoint/baseline_3_Checkpoint/finetuned_yolov4_tiny_epoch_6.pth",
#                "/content/baseline_checkpoint/baseline_3_Checkpoint/finetuned_yolov4_tiny_epoch_12.pth",
#                "/content/baseline_checkpoint/baseline_3_Checkpoint/finetuned_yolov4_tiny_epoch_18.pth",
#                "/content/baseline_checkpoint/baseline_3_Checkpoint/finetuned_yolov4_tiny_epoch_24.pth",
#                "/content/baseline_checkpoint/baseline_3_Checkpoint/finetuned_yolov4_tiny_epoch_30.pth",
#                ]

model_files = ["/content/baseline_checkpoint/baseline_3_Experiment_2_Checkpoint/finetuned_yolov4_tiny_epoch_4.pth",
               "/content/baseline_checkpoint/baseline_3_Experiment_2_Checkpoint/finetuned_yolov4_tiny_epoch_8.pth",
               "/content/baseline_checkpoint/baseline_3_Experiment_2_Checkpoint/finetuned_yolov4_tiny_epoch_12.pth",
               "/content/baseline_checkpoint/baseline_3_Experiment_2_Checkpoint/finetuned_yolov4_tiny_epoch_16.pth",
               "/content/baseline_checkpoint/baseline_3_Experiment_2_Checkpoint/finetuned_yolov4_tiny_epoch_20.pth",
                ]

# epoch_no = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
# epoch_no = [6,12,18,24,30]
epoch_no = [4,8,12,16,20]

# Evaluate all models
results = []
for model_file in model_files:
    print(f"\nEvaluating {model_file}")

    # Load model
    model = YOLOv4Tiny(cfg_file, num_classes=num_classes, img_size=img_size)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    # Evaluate
    mAP, precision, recall, f1 = evaluate_model(model, test_loader, device, num_classes)
    results.append({
        'epoch': epoch_no,
        # 'epoch': [4,8,12,16,20],
        # 'epoch': int(re.search(r'epoch_(\d+)', model_file).group(1)),
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    # Free ram
    del model
    torch.cuda.empty_cache()

# Plot graph
plt.figure(figsize=(15, 10))

# mAP
# plt.subplot(2, 2, 1)
plt.figure(figsize=(15, 6))
plt.plot(epoch_no, [r['mAP'] for r in results], 'b-o')
plt.title('mAP vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.grid(True)
plt.savefig('mAP_Plot.png')

# Precision
plt.figure(figsize=(15, 6))
plt.subplot(3, 1, 1)
plt.plot(epoch_no, [r['precision'] for r in results], 'r-o')
plt.title('Precision vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim(0, 1)
plt.grid(True)

# Recall
plt.subplot(3, 1, 2)
plt.plot(epoch_no, [r['recall'] for r in results], 'g-o')
plt.title('Recall vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim(0, 1)
plt.grid(True)

# F1 Score
plt.subplot(3, 1, 3)
plt.plot(epoch_no, [r['f1'] for r in results], 'm-o')
plt.title('F1 Score vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.savefig('Precision_Recall_F1_Score_Plot.png')
plt.show()

print("\nEvaluation Results:")
print("Epoch | mAP   | Precision | Recall | F1 Score")
print("----------------------------------------------")
for i in range(len(results)):
    res = results[i]
    # Handling the epoch field
    # epoch = int(res['epoch'])
    epoch = epoch_no

    # Handling others field
    mAP = res['mAP'].item() if isinstance(res['mAP'], np.ndarray) else res['mAP']
    precision = res['precision'].item() if isinstance(res['precision'], np.ndarray) else res['precision']
    recall = res['recall'].item() if isinstance(res['recall'], np.ndarray) else res['recall']
    f1 = res['f1'].item() if isinstance(res['f1'], np.ndarray) else res['f1']

    print(f"{epoch[i]} | {mAP:.3f} | {precision:.3f}   | {recall:.3f} | {f1:.3f}")
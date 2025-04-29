import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorchyolov4tiny.models import YOLOv4Tiny, YOLOLayer


def xywh_to_xyxy(boxes):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=1)

def predict_boxes(output_image_path, final_detections):
    # Convert model input tensor back to PIL Image
    input_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_image = (input_image * 255).astype(np.uint8)
    base_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(base_image)

    # Define colors and font
    colors = plt.cm.get_cmap('hsv', num_classes) 
    font = ImageFont.load_default()

    # Draw each detection
    for det in final_detections.cpu().numpy():
        x_center, y_center, width, height = det[:4]
        conf = det[4]
        class_id = int(det[5])

        # Convert from center coordinates to corner coordinates
        x1 = x_center - width/2
        y1 = y_center - height/2
        x2 = x_center + width/2
        y2 = y_center + height/2

        # Get color for this class
        color = tuple(int(255 * x) for x in colors(class_id/num_classes)[:3])

        # Draw rectangle and label
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"Class {class_id} ({conf:.2f})"
        text_size = draw.textbbox((0,0), label, font=font)
        draw.rectangle([x1, y1, x1+text_size[2], y1+text_size[3]], fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

    # Save and display
    base_image.save(output_image_path)
    print(f"\nAnnotated image saved to {output_image_path}")

    # Display using matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(base_image)
    plt.axis('off')
    plt.title("YOLOv4-tiny Predictions")
    plt.show()
    
def detect_and_save(image_path, output_txt_path, output_image_path, model_path, cfg_file, num_classes, img_size, conf_threshold, iou_threshold, top_conf):
    """
    Perform object detection on an image using YOLOv4 Tiny and save results to a text file.
    
    Args:
        image_path (str): Path to the input image
        output_txt_path (str): Path to save the detection results
        model_path (str): Path to the pretrained model weights
        cfg_file (str): Path to the model configuration file
        num_classes (int): Number of classes in the model
        img_size (int): Size to resize the image (img_size x img_size)
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
        top_conf (int): Maximum number of detections to keep
    """
    # Set device
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
        predictions = model(image_tensor)
    
    # Concatenate predictions
    detections = torch.cat(predictions, dim=1)[0]
    
    # Process detections
    boxes = detections[:, :4]  # [x, y, w, h] in pixel coordinates
    objectness = detections[:, 4]
    class_scores = detections[:, 5:]
    max_class_score, class_id = torch.max(class_scores, dim=1)
    conf = objectness * max_class_score
    
    # Filter detections
    mask = conf > conf_threshold
    selected_boxes = boxes[mask]
    selected_conf = conf[mask]
    selected_class_id = class_id[mask]
    selected_class_scores = class_scores[mask]
    
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
    
    # Print detections
    print(f"Detections for {image_path}:")
    if final_detections.size(0) > 0:
        for i in range(final_detections.size(0)):
            x, y, w, h = final_detections[i, :4]
            conf = final_detections[i, 4]
            class_id = final_detections[i, 5].long()
            print(f"Class: {class_id.item()}, Confidence: {conf.item():.4f}, "
                  f"Box: [x={x.item():.2f}, y={y.item():.2f}, w={w.item():.2f}, h={h.item():.2f}]")
        predict_boxes(output_image_path, final_detections)
    else:
        print("No detections found.")
    print(f"Detections saved to {output_txt_path}")

if __name__ == "__main__":
    import yaml
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    detect_and_save("./input.png", "./detections_output.txt", "./output.jpg", "../weights/baseline_model_epoch_30.pth", 
                "../config/yolov4-tiny.cfg", config['num_classes'], config['img_size'], 
                config['conf_threshold'], config['iou_threshold'], config['top_conf'])
  

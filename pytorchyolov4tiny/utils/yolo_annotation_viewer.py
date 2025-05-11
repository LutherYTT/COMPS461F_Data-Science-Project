# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# Draw bounding boxes and labels on the image
def draw_labels(image, labels):
    height, width, _ = image.shape
    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.split())

        # Convert YOLO format label to frame coordinates
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        bbox_width = int(bbox_width * width)
        bbox_height = int(bbox_height * height)

        # Calculate top-left corner coordinates of the bounding box
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Class {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Browse images using keyboard.
# A: Previous image, D: Next image, ESC: Exit
def visualize_dataset(image_dir, label_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    index = 0
    total_images = len(image_files)

    while True:
        image_file = image_files[index]
        label_file = image_file.rsplit('.', 1)[0] + '.txt'

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        # Load image and labels
        image = cv2.imread(image_path)
        try:
            with open(label_path, 'r') as f:
                labels = f.readlines()
        except FileNotFoundError:
            print(f"Label file for {image_file} not found. Displaying image without labels.")
            labels = []

        # Draw labels on the image
        labeled_image = draw_labels(image.copy(), labels)

        # Display the image
        cv2.namedWindow("Labeled Image", cv2.WINDOW_NORMAL)
        cv2.imshow('Labeled Image', labeled_image)
        cv2.resizeWindow('Labeled Image', 700, 700)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC key to exit
            break
        elif key == ord('a') or key == 81:  # 'A' key to show previous image
            index = (index - 1) % total_images
        elif key == ord('d') or key == 83:  # 'D' key to show next image
            index = (index + 1) % total_images

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_dir = './DataSynthetics/output/images'
    label_dir = './DataSynthetics/output/labels_fine'
    visualize_dataset(image_dir, label_dir)

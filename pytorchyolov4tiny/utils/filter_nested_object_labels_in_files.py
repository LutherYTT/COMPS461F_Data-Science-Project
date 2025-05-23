# -*- coding: utf-8 -*-
import os

# Parse each label line to get x, y, width, height for further checking
def parse_box(line):
    parts = line.strip().split()
    return {
        'class_id': int(parts[0]),
        'x_center': float(parts[1]),
        'y_center': float(parts[2]),
        'width': float(parts[3]),
        'height': float(parts[4]),
        'line': line  # Keep the original label line for writing back
    }

# Check if box2 is strictly inside box1 (completely covered)
def is_strictly_inside(box1, box2):
    # Calculate box1 boundaries
    b1_left = box1['x_center'] - box1['width'] / 2
    b1_right = box1['x_center'] + box1['width'] / 2
    b1_top = box1['y_center'] - box1['height'] / 2
    b1_bottom = box1['y_center'] + box1['height'] / 2

    # Calculate box2 boundaries
    b2_left = box2['x_center'] - box2['width'] / 2
    b2_right = box2['x_center'] + box2['width'] / 2
    b2_top = box2['y_center'] - box2['height'] / 2
    b2_bottom = box2['y_center'] + box2['height'] / 2

    # Compare all four edges to check if box2 is completely inside box1
    return (b1_left < b2_left and b1_right > b2_right and
            b1_top < b2_top and b1_bottom > b2_bottom)

def process_label_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse x, y, width, height from each line
    boxes = [parse_box(line) for line in lines if line.strip()]  # Ignore empty lines

    # Compare each box with every other box to find invalid labels
    to_remove = set()
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i != j and is_strictly_inside(boxes[i], boxes[j]):
                to_remove.add(j)  # Mark label for removal
                break  # Once box j is marked, no need to compare further with i

    # Keep only the labels that are not marked for removal
    filtered_boxes = [boxes[i]['line'] for i in range(len(boxes)) if i not in to_remove]

    with open(filepath, 'w') as f:
        for line in filtered_boxes:
            f.write(line)
if __name__ == "__main__":
    label_folder = ".\output\labels_fine"
    
    for filename in os.listdir(label_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_folder, filename)
            process_label_file(filepath)
            print(f"Processed {filename}")

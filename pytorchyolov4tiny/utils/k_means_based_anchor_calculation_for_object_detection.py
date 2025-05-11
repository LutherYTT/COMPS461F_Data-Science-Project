# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
import os

def compute_anchors(annotation_files, num_anchors=9, img_size=832):
    boxes = []
    for ann_file in annotation_files:
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    boxes.append([w * img_size, h * img_size])
    boxes = np.array(boxes)
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    return anchors

if __name__ == "__main__":
    train_list_file = '/content/train.txt'
    img_size=832
    
    # Collect annotation files
    with open(train_list_file, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]
    annotation_files = [os.path.splitext(img_path)[0] + '.txt' for img_path in img_paths]
    anchors = compute_anchors(annotation_files, num_anchors=9, img_size=img_size)
    print("Computed anchors:", anchors)

# **Integrated Food Object Detection and Nutrition-Based Recommender System**

## **Table of Contents**
- [Team Members](#team-members)
- [Supervisor](#supervisor)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Introduction](#introduction)
- [System Overview](#system-overview)
- [Utilities](#utilities)
  - [Object Detection Utilities](#object-detection-utilities)
  - [Recommender System Utilities](#recommender-system-utilities)
- [Object Detection (YOLOv4-Tiny)](#object-detection-yolov4-tiny)
- [Recommender System](#recommender-system)
- [Results and Evaluation](#results-and-evaluation)
  - [Object Detection](#object-detection-results)
  - [Recommender System](#recommender-system-results)  
  - [Analysis of Evaluation Results](#analysis-of-evaluation-results)
- [Future Work](#future-work)


## **Team Members**
| **Name**         | **Email**              |
|-------------------|------------------------|
| Ng Sing Man      | simon78451213@gmail.com |
| Kwok Wai Chun    | shindo.light@gmail.com |
| Ng Chiu Cheuk    | brian17659@gmail.com  |
| Chu Sik Hin      | acalvin701@gmail.com  |
| Chow Chun Ting   | jaychow603@gmail.com  |
| Lu Yuk Tong      | luyuktong@gmail.com   |

## **Supervisor**
**Mr. Jimmy Kang**  
Email: [xkang@hkmu.edu.hk](mailto:xkang@hkmu.edu.hk)

## **Datasets**
| **Dataset**      | **Description**                                                                 | **Link**                                                                                       |
|-------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Training Data    | 16,000 synthetic images with annotations for food items and nutrition labels.   | [Download](https://drive.google.com/file/d/1HwEnkcGAbojyMBAgjtRsdpl2rs51Th5U/view?usp=sharing) |
| Validation Data  | 4,000 synthetic images used for model validation.                              | [Download](https://drive.google.com/file/d/1BzfwQlHQfW_jAZRO3ChEZ9s40h-5y5FD/view?usp=sharing) |
| Test Data        | 4,000 images with new backgrounds, miscellaneous items, and larger transformations (150% of original size) for robust testing. | [Download](https://drive.google.com/file/d/1Dmv40IaUDQy5ZXNWBbU7BXkPswlhlx7J/view?usp=sharing) |


## **Installation**
To install this Project, follow these steps:
1. Clone the repository: **`git clone [https://github.com/username/project-title.git](https://github.com/LutherYTT/COMPS461F_Data-Science-Project.git)`**
2. Navigate to the project directory: **`cd COMPS461F_Data-Science-Project`**
3. Install requirements.txt: **`pip install -r requirements.txt`**
4. Run main.py: **`python main.py`**


## **Usage**
**Example Outputs**:  
- Sample Input:  
  ![Sample Input Image](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/images/input.png)  
- Detection Output:  
  ![Detection Output](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/images/output.jpg)


## **Introduction**
This project aims to assist individuals with dietary sensitivities and health concerns by combining object detection and a recommender system. The system detects food items and nutrition labels in images or videos using **YOLOv4-tiny** and provides personalized food recommendations based on nutritional needs and user preferences. It addresses challenges in integrating these technologies to offer practical dietary guidance, helping users maintain healthier lifestyles.

The system is inspired by real-world applications, such as food recognition software at 7-Eleven stores in Hong Kong, and leverages:
- **YOLOv4-tiny**: A lightweight, efficient object detection model.
- **Hybrid Recommender System**: Combines collaborative filtering (user behavior) and content-based filtering (item attributes) for accurate suggestions.

## **System Overview**
The system integrates two core components:
1. **Object Detection**: Identifies food items and nutrition labels.
2. **Recommender System**: Suggests personalized food options.

![System Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/whole_architecture.drawio.png)


## **Utilities**
### **Object Detection Utilities**
| **Index** | **Name**                                   | **Description**                                                                                     |
|-----------|--------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 1         | `video_frame_extractor`                   | Extracts frames from videos at regular intervals.                                                  |
| 2         | `batch_background_removal_and_cropping`   | Removes backgrounds and crops images to focus on objects, saving them as PNGs.                     |
| 3         | `synthetic_image_generation_with_random_transformations` | Generates synthetic images with random transformations for training.                  |
| 4         | `filter_nested_object_labels_in_files`    | Removes invalid nested bounding boxes from YOLO annotations.                                       |
| 5         | `yolo_annotation_viewer`                  | Visualizes images with YOLO bounding boxes, navigable via keyboard.                                |
| 6         | `k_means_based_anchor_calculation_for_object_detection` | Calculates optimized anchor box sizes using KMeans clustering.                       |

### **Recommender System Utilities**
| **Index** | **Name**                                   | **Description**                                                                                     |
|-----------|--------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 1         | `rule_based_food_combination_generator`   | Generates synthetic food combinations based on predefined rules, saved as CSV.                     |
| 2         | `generate_nutritional_combinations_of_foods` | Creates all possible food combinations (1-5 items) with summed nutritional values.              |
| 3         | `association_rules_mining`                | Mines association rules from synthetic data using FP-Growth, saved as CSV.                         |
| 4         | `food_combination_snr_calculator`         | Calculates signal-to-noise ratio to evaluate combination quality.                                  |
| 5         | `weight_tuning_for_cosine_and_association_scores` | Tunes weights for combining cosine similarity and association scores.                      |
| 6         | `evaluation_coverage`                     | Measures recommendation coverage against the product dataset.                                      |
| 7         | `plot_association_rule_confidence_distribution` | Plots confidence score distributions for association rules across noise levels.              |


## **Object Detection (YOLOv4-Tiny)**
We use **YOLOv4-tiny** for its speed and efficiency, ideal for real-time detection. Its architecture includes:
- Convolutional layers for feature extraction.
- Route layers for multi-scale feature integration.
- Two YOLO layers for detecting objects at different scales.

**Why YOLOv4-tiny?** It offers a lightweight alternative to larger models like YOLOv4, balancing accuracy and performance on resource-constrained devices.

![YOLO Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Yolo%20architecture%20fixed.drawio.png)  
![Detection Workflow](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Detection.drawio.png)


### **Process**
1. **3D Model Generation**: Use SUDOAI to create 3D food models and record rotations.
2. **Frame Extraction**: Extract frames from 3D model rotations.
3. **Preprocessing**: Remove backgrounds and crop excess space.
4. **Image Synthesis**: Apply random transformations (e.g., rotation, blur) for training data.
5. **Label Validation**: Discard overlapping or invalid bounding boxes.

![Rotation 3D model](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/model%20generated%20by%20sudoai.gif)
![Object Detection Process](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Object_Detection_Preprocess.drawio.png)

### **Image Synthesis**
Synthetic images are created by adding random backgrounds and transformations to preprocessed images, ensuring robustness. Overlaps are managed by validating labels to avoid occlusion issues.

![Image Synthesis](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/data%20synthesis.drawio.png)
![Object Overlapping Problem](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Image_Generate_Covering.drawio.png)

## **Recommender System**
The recommender system delivers personalized food suggestions using:
- **Cosine Similarity**: Matches nutrients quickly via KD-tree.
- **Association Rules**: Mined with FP-Growth from synthetic transaction data.

Weights between these methods are tuned to optimize recommendation quality.

![Recommender System Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/recommender_system_architecture.png)


### **Food Combination Generation**
Synthetic transaction data is generated with rules:
1. No repeated food types.
2. Meat/seafood paired with vegetables.
3. One staple food (e.g., rice) per combination.
4. Maximum one drink.
5. Sandwiches and wraps never co-occur.

![Food Combination Generation](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/synthetic_Transaction.drawio.png)


### **Hyperparameter Tuning**
Weights for cosine similarity and association rules are optimized by minimizing a loss function based on score overlap, ensuring relevant recommendations.

![Hyperparameter Tuning](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Hyperparameter_Tuning.drawio.png)


## **Results and Evaluation**
### **Object Detection Results**
In our object detection experiments, we assessed model performance using **F1-score**, **Recall**, **Precision**, and **mean Average Precision (mAP)**. The following sections detail the results and analyses from three distinct experiments with varying training configurations.
#### Experiment 1: Learning Rate = 0.001
**Evaluation Results:**
| Epoch | mAP   | Precision | Recall | F1 Score |
|-------|-------|-----------|--------|----------|
| 4     | 0.228 | 0.815     | 0.225  | 0.353    |
| 8     | 0.237 | 0.840     | 0.160  | 0.268    |
| 12    | 0.278 | 0.789     | 0.339  | 0.474    |
| 16    | 0.287 | 0.632     | 0.437  | 0.517    |
| 20    | 0.400 | 0.389     | 0.527  | 0.448    |

**Analysis:**  
With a learning rate of 0.001, the model’s learning accelerated, but precision, recall, and F1-score declined over epochs. Meanwhile, mAP increased, suggesting improved object detection despite the trade-off. This indicates faster convergence, potentially leading to overfitting, as the model struggles to generalize, lowering precision and recall.
#### Experiment 2: SGD with Momentum and Weight Decay
**Evaluation Results:**
| Epoch | mAP   | Precision | Recall | F1 Score |
|-------|-------|-----------|--------|----------|
| 4     | 0.368 | 0.815     | 0.446  | 0.577    |
| 8     | 0.384 | 0.681     | 0.495  | 0.573    |
| 12    | 0.382 | 0.799     | 0.467  | 0.589    |
| 16    | 0.339 | 0.582     | 0.471  | 0.521    |
| 20    | 0.370 | 0.856     | 0.390  | 0.536    |

**Analysis:**  
Switching to Stochastic Gradient Descent (SGD) with momentum and weight decay resulted in fluctuating precision, recall, and F1-scores, with inconsistent mAP. These variations stem from SGD’s sensitivity to data noise, causing unstable updates and inconsistent performance as the model adapts to noisy samples.
#### Experiment 3: Learning Rate Schedule
**Evaluation Results:**
| Epoch | mAP   | Precision | Recall | F1 Score |
|-------|-------|-----------|--------|----------|
| 4     | 0.211 | 0.864     | 0.173  | 0.288    |
| 8     | 0.258 | 0.830     | 0.299  | 0.439    |
| 12    | 0.284 | 0.692     | 0.405  | 0.511    |
| 16    | 0.254 | 0.774     | 0.330  | 0.463    |
| 20    | 0.360 | 0.591     | 0.456  | 0.515    |

**Analysis:**  
A learning rate schedule stabilized precision, recall, and F1-scores by avoiding disruptive updates. Although mAP fluctuated, it showed an upward trend, reflecting consistent improvement in detection performance over time.

### **Recommender System Results**
#### Impact of Noise on Model Confidence
![Confidence Distributions Plot](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/evaluate_graph/(recommender_system)confidence_distributions_lines.png)
This section examines how noise levels in a synthetic dataset affect the model's confidence, visualized through a line chart. The chart separates confidence into coarse and fine class categories.

- **Key Findings:**
  - Higher noise levels reduce the model's confidence.
  - Confidence decreases as noise increases, observed across both coarse and fine classes.

These results emphasize the need to control noise to ensure reliable model predictions.

#### Recommender System Coverage
Coverage was assessed using outputs from 100 object detections, measuring the proportion of food items recommended by the system.

- **Results:**
  - **Coverage:** 50% (only half of the food items are recommended).
  - **Context:** This is acceptable given the system’s focus on addressing nutritional gaps and the limited variety of available food options.

While not optimal, the 50% coverage reflects a practical balance between relevance and the constraints of the dataset.


### **Analysis of Evaluation Results**
The following table presents the evaluation metrics for our object detection model at various training epochs:

| Epoch | mAP   | Precision | Recall | F1 Score |
|-------|-------|-----------|--------|----------|
| 4     | 0.368 | 0.815     | 0.446  | 0.577    |
| 8     | 0.384 | 0.681     | 0.495  | 0.573    |
| 12    | 0.382 | 0.799     | 0.467  | 0.589    |
| 16    | 0.339 | 0.582     | 0.471  | 0.521    |
| 20    | 0.370 | 0.856     | 0.390  | 0.536    |

Upon analyzing these results, we observe that the precision is consistently higher than the mean Average Precision (mAP), Recall, and F1 score across all epochs. This discrepancy arises due to the characteristics of the synthetic dataset used for training and evaluation. The background images selected for image synthesis contain a large number of miscellaneous objects that were not labeled as part of the target classes. Since the labels were added during the synthesis process and did not account for these pre-existing background objects, the model faces difficulty detecting all labeled objects accurately. This leads to missed detections, resulting in lower recall. However, when the model does make a detection, it is typically correct, which contributes to the high precision. The presence of these unlabeled miscellaneous objects primarily impacts the recall, consequently lowering the mAP and F1 score, which are more sensitive to missed detections.

To address this issue and enhance the model's overall performance, we propose a solution: during the image synthesis process, use background images that contain fewer miscellaneous objects. This adjustment will allow the model to better focus on detecting the labeled objects, thereby improving metrics such as recall, mAP, and F1 score.


## **Future Work**
- Incorporate real-world images into the dataset.
- Explore advanced models like YOLOv5 or EfficientDet.
- Add user feedback to improve recommendations.

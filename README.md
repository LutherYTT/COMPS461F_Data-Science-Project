# **Food Nutrition Label Detection and Recommendation System**

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

![sample input image](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/images/input.png)
![detection output](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/images/output.jpg)

## **Introduction**
The aim of this project is to address the problem of supporting individuals who have diet sensitivity and health-related issues by developing a combination of technological integration with a health awareness approach. The main objectives include focusing on solving the challenges caused by integrating object detection systems into recommendation systems to provide personal food guidance, formulating an enhanced formula and algorithm capable of recommending the best choice of diet for its end-users, and thereby facilitating users in maintaining controlled diets and a healthy physique. 

At 7-Eleven stores in Hong Kong, food recognition software identifies items, helping consumers choose healthier options. This technology utilizes Artificial Intelligence, which employs sophisticated algorithms that can detect and locate objects in images or videos. One instantiation of this technology is the YOLO v4 tiny algorithm. Recommended systems are driven by data science. Hybrid methods use collaborative filtering (which considers user behavior) with content-based filtering (which considers item attributes). These tend to yield accurate suggestions. Knowledge-based systems use explicit health-based rules to provide recommendations. 

## **System Overview**
![Whole System Architecture Graph](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/whole_architecture.drawio.png)

## **Utils**
### Object Detection
| **Index** | **Name** | **Description** | 
| -- | -- | -- |
| 1 | video_frame_extractor | Extracts and saves frames from a video at regular time intervals into an output folder. |
| 2 | batch_background_removal_and_cropping | Removes the background from images in a folder, trims the resulting images to remove extra blank space, converts them to PNG format, and saves them to an output folder. |
| 3 | synthetic_image_generation_with_random_transformations | Generates synthetic images by randomly composing food and miscellaneous items onto background images with various random transformations, and creates corresponding fine and coarse object detection labels for training machine learning models. | 
| 4 | filter_nested_object_labels_in_files | Removes label boxes that are completely inside other boxes from YOLO-format annotation files in a folder. |
| 5 | yolo_annotation_viewer | Visualizes images with their YOLO-format bounding box labels, allowing navigation through the dataset using keyboard keys. |
| 6 | k_means_based_anchor_calculation_for_object_detection | Computes optimized anchor box sizes for object detection by clustering bounding box dimensions from annotation files using KMeans. |

### Recommender System
| **Index** | **Name** | **Description** | 
| -- | -- | -- |
| 1 | rule_based_food_combination_generator | Generates random food category combinations with specific probability rules and assigns product names, then saves the results to a CSV file. |
| 2 | generate_nutritional_combinations_of_foods | Generates and saves all possible food item combinations (1 to 5 items) with summed nutritional values from a product CSV file. |
| 3 | association_rules_mining | Mines frequent itemsets and generates association rules with weighted support and confidence from food combination data, then saves the rules to a CSV file. |
| 4 | food_combination_snr_calculator | Evaluates food combinations against rules, counts violations as noise, and calculates the signal-to-noise ratio (SNR) to measure data quality. |
| 5 | weight_tuning_for_cosine_and_association_scores | Finds the best pair of weights to minimize pairwise ranking loss based on cosine similarity and association scores. |
| 6 | evaluation_coverage | Evaluates recommendation results by processing prediction files, filtering out irrelevant data, combining scores, and calculating the coverage of unique recommended items against the total product dataset. |
| 7 | plot_association_rule_confidence_distribution | Plots and compares the distributions of coarse and fine class confidence scores from multiple CSV files representing different noise levels. |

## **Object Detection(Yolov4 Tiny)**
Our object detection uses YOLOv4-tiny for its fast and efficient balance of accuracy and speed, ideal for real-time tasks. It features a simplified architecture with convolutional layers for feature extraction, route layers for multi-scale integration, and two YOLO layers to predict bounding boxes and classes at different scales, enabling effective detection of various object sizes.
![Yolo Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Yolo%20architecture%20fixed.drawio.png)
![Detection Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Detection.drawio.png)

### Object Detection(Process)
Beginning with a raw image that has gone through a generative AI model (SUDOAI) for 3D model generation and recorded the rotations of the 3D model, which are then extracted as different frames, which are pre-processed through background removal and cropping excess white space to get different perspectives of that object, this pre-processed image will be used to synthesize the image for model training.
![Rotation 3D model](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/model%20generated%20by%20sudoai.gif)
![Object Detection Process Method Design](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Object_Detection_Preprocess.drawio.png)

### Object Detection(Image Synthesis)
The pre-processed images are synthesized by applying random backgrounds and transformations (rotation, blur, flip, scaling, positioning, brightness, contrast) in random order to create the dataset. To handle near-far constraints and occlusions, we use a label validation step that discards invalid labels where one object's bounding box is fully inside another, ensuring objects and backgrounds do not improperly overlap.
![Image Synthesis Method Design](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/data%20synthesis.drawio.png)
![Object Overlapping Problem](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Image_Generate_Covering.drawio.png)

## **Recommender System**
Our recommender system is designed to provide personalized food combinations that meet users' nutritional needs while minimizing reliance on explicit user information. It combines two key methods: cosine similarity, using a KD-tree for fast nutrient matching, and association rules mined via FP-Growth from synthetic eating habit data. By tuning weights between these methods, the system delivers relevant recommendations that address nutritional gaps while reducing unsuitable options.
![Recommender System Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/recommender_system_architecture.png)

### Recommender System (Food Combination)
For our recommender system, we generated synthetic transaction data simulating eating habits to ensure diverse association rules. We applied five rules: 1. no repeated food types in a combination; 2. each must include meat or seafood paired with vegetables; 3. only one staple food (rice/noodles) allowed; 4. at most one drink per combination; and 5. sandwiches and wraps never appear together.
![Food Combination Generation Design](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/synthetic_Transaction.drawio.png)

### Recommender System(Hyperparameter Tuning)
Our recommender system combines cosine similarity and association rules, each weighted to produce a final ranking score. For hyperparameter tuning, we randomly generate cosine similarity scores and weights, then calculate a loss based on overlapping final scores. By minimizing this loss, we optimize weights to recommend food combinations that effectively address nutritional gaps while avoiding unsuitable suggestions.
![Hyperparameter Tuning Method Design](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Hyperparameter_Tuning.drawio.png)


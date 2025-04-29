# **Object Detection of Food Nutrition Label**

## **Member**
| Name | Email |
| --- | --- |
| Ng Sing Man | simon78451213@gmail.com |
| Kwok Wai Chun | shindo.light@gmail.com |
| Ng Chiu Cheuk | brian17659@gmail.com |
| Chu Sik Hin | acalvin701@gmail.com |
| Chow Chun Ting | jaychow603@gmail.com |
| Lu Yuk Tong | luyuktong@gmail.com |

## **Supervisor**
### **Mr. Jimmy KANG (xkang@hkmu.edu.hk)**

## **Dataset**
| Dataset | Description | File | 
| -- | -- | -- |
| Training Data | 16000 images | https://drive.google.com/file/d/1HwEnkcGAbojyMBAgjtRsdpl2rs51Th5U/view?usp=sharing |
| Validation Data | 4000 images | https://drive.google.com/file/d/1BzfwQlHQfW_jAZRO3ChEZ9s40h-5y5FD/view?usp=sharing |
| Test Data | 4000 images, with new background and misc images, and using larger random transformation | https://drive.google.com/file/d/1Dmv40IaUDQy5ZXNWBbU7BXkPswlhlx7J/view?usp=sharing |

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

## **Object Detection(Yolov4 Tiny)**
![Yolo Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Yolo%20architecture%20fixed.drawio.png)
![Detection Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/Detection.drawio.png)
Our object detection component is powered by the YOLOv4-tiny model, a compact and efficient version of the YOLO (You Only Look Once) real-time object detection framework. We chose YOLOv4-tiny for its optimal trade-off between detection accuracy and processing speed, making it ideal for applications requiring fast analysis of visual data, such as identifying objects in real-time scenarios. The model employs a simplified architecture with convolutional layers for feature extraction, route layers for multi-scale feature integration, and dual YOLO layers to predict bounding boxes and class probabilities across different object sizes. This setup ensures effective detection of diverse objects in input images.

## Recommender System
![Recommender System Architecture](https://github.com/LutherYTT/COMPS461F_Data-Science-Project/blob/main/assets/architecture_graph/recommender_system_architecture.png)
Our recommender system is designed to provide personalized food combinations that meet users' nutritional needs while minimizing reliance on explicit user information. The methodology consists of two main components cosine similarity and association rules. In the Cosine Similarity component, we efficiently identify food combinations based on the nutrients requiring supplementation by utilizing a KD-tree structure. This approach significantly reduces the computational burden associated with exhaustive similarity calculations. We then compute similarity scores to assess the proximity of food combinations to the target nutrients. In the Association Rules component, we employ the FP-Growth algorithm to mine patterns from synthetic transaction data. To generate this synthetic data, we establish rules that simulate eating habits. 

Finally, we conduct hyperparameter tuning by randomly assigning weights to both cosine similarities and association rules. The final recommendation score is derived from a weighted combination of these metrics. This structured approach ensures that our system delivers relevant food combinations that effectively address nutritional gaps while minimizing unsuitable suggestions. 

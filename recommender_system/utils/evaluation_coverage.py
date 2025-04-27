import pandas as pd
import glob
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recommender_system import *

import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '../..', 'config.yml')
config_path = os.path.normpath(config_path)
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def calculate_coverage(unique_recommended_items, total_count):
    coverage = len(unique_recommended_items) / total_count
    return coverage

def load_total_items(csv_file):
    count_df = pd.read_csv(csv_file)
    total_items = len(count_df)
    return total_items

def process_prediction_file(prediction_txt, nutrition_dataset):
    # Read the prediction file and check for all classes being 12
    with open(prediction_txt, 'r') as file:
        rows = [list(map(float, line.split())) for line in file.readlines()]

    # Check if all classes are 12 (misc)
    if all(int(row[0]) == 12 for row in rows):
        print(f"Skipping {prediction_txt}: All classes are 'misc'.")
        return None, []  # Skip processing this file

    # Filter out "misc" rows
    rows = [row for row in rows if int(row[0]) != 12]

    coarse_prob, fine_prob, predicted_item = sum_probability(prediction_txt)
    
    # Calculate nutrition gap
    nutrition_gap = calculate_nutrition_gap(coarse_prob, fine_prob, nutrition_dataset)
    
    preprocess_csv = "../../datasets/7select_Product_preprocess.csv"
    association_rule_csv = "../../datasets/association_rule/association_rule_noise_ratio_01.csv"
    df = combined_score(nutrition_gap, coarse_prob, predicted_item, association_rule_csv, preprocess_csv)
    return df, predicted_item

# Main evaluation
nutrition_dataset = "../../datasets/7select_Product.csv"
total_products_count = load_total_items(nutrition_dataset)
all_unique_recommended_dishes = set()
skipped_files_count = 0

all_detection_files = glob.glob('../../datasets/sample_detections_output/synthetic_*_detections.txt')

for prediction_txt in all_detection_files:
    result, predicted_item = process_prediction_file(prediction_txt, nutrition_dataset)
    if result is None:  # If the result is None,skip that file
        skipped_files_count += 1
    else:
        # Extract unique dishes from the result
        unique_recommended_dishes = set()
        if 'Dish' in result.columns:
            for dish in result['Dish'].head(3):  # Only consider the top 3 rows
                items = [item.strip() for item in dish.split(',')]
                unique_recommended_dishes.update(items)
                
        # Update the overall unique recommended dishes
        all_unique_recommended_dishes.update(unique_recommended_dishes)
        
# Calculate coverage
coverage = calculate_coverage(all_unique_recommended_dishes, total_products_count)
# Output coverage and number of skipped files
print(f"Number of skipped files: {skipped_files_count}")
print(f"Coverage: {coverage:.2f}")

from recommender_system.recommender_system import *
from pytorchyolov4tiny.detect import detect_and_save

if __name__ == "__main__":
    import yaml
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    detect_and_save("images/input.png", "./detections_output.txt", config['model_path'], 
                config['cfg_file_path'], config['num_classes'], config['img_size'], 
                config['conf_threshold'], config['iou_threshold'], config['top_conf'])
    
    prediction_txt = "./detections_output.txt"
    nutrition_dataset = config['nutrition_dataset']
    preprocess_csv = config['preprocess_csv'] 
    association_rule_csv = config['association_rule_csv']

    # Integration probability
    coarse_prob, fine_prob, predicted_item = sum_probability(prediction_txt)
    
    # Nutrition gap
    nutrition_gap = calculate_nutrition_gap(coarse_prob, fine_prob, nutrition_dataset)
    
    df = combined_score(nutrition_gap, coarse_prob, predicted_item, association_rule_csv, preprocess_csv)

    output_df = df[['Dish', 'Final_Score']].head(10)
    output_df = output_df.rename(columns={'Dish': 'Food Combination ', 'Final_Score': 'Recommended Index'})

    print("We recommend that you add the following:")

    print(output_df.to_string(index=False))
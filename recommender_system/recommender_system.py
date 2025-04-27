import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os

def sum_probability(prediction_txt):
    with open(prediction_txt, 'r') as file:
        rows = [list(map(float, line.split())) for line in file.readlines()]

    predicted_item = []

    coarse_prob = {
        "coarse_breast": [],
        "coarse_sandwich": [],
        "coarse_wrap": []
    }

    fine_prob = {
        "basil_garlic_chicken_breast": [],
        "basil_garlic_chicken_breast_2": [],
        "black_pepper_chicken_breast": [],
        "herbs_chicken_breast": [],
        "bacon_egg_mayo_sandwich": [],
        "ham_cheese_pickle_sandwich": [],
        "smoked_salmon_egg_mayo_sandwich": [],
        "super_club_sandwich": [],
        "caesar_chicken_cold_wrap": [],
        "peking_duck_cold_wrap": [],
        "sesame_chicken_with_baby_spinach_cold_wrap": [],
        "smoked_salmon_egg_salad_cold_wrap": []
    }

    for row in rows:
        temp = int(row[0])
        # if item is not misc
        if (temp != 12):
            predicted_item.append(list(fine_prob.keys())[temp])

            # Store the probability of each of the coarse class and fine class into the
            coarse_prob["coarse_breast"].append(row[2]+row[3]+row[4]+row[5])
            coarse_prob["coarse_sandwich"].append(row[6]+row[7]+row[8]+row[9])
            coarse_prob["coarse_wrap"].append(row[10]+row[11]+row[12]+row[13])

            fine_prob["basil_garlic_chicken_breast"].append(row[2])
            fine_prob["basil_garlic_chicken_breast_2"].append(row[3])
            fine_prob["black_pepper_chicken_breast"].append(row[4])
            fine_prob["herbs_chicken_breast"].append(row[5])
            fine_prob["bacon_egg_mayo_sandwich"].append(row[6])
            fine_prob["ham_cheese_pickle_sandwich"].append(row[7])
            fine_prob["smoked_salmon_egg_mayo_sandwich"].append(row[8])
            fine_prob["super_club_sandwich"].append(row[9])
            fine_prob["caesar_chicken_cold_wrap"].append(row[10])
            fine_prob["peking_duck_cold_wrap"].append(row[11])
            fine_prob["sesame_chicken_with_baby_spinach_cold_wrap"].append(row[12])
            fine_prob["smoked_salmon_egg_salad_cold_wrap"].append(row[13])

    print("All coarse class and fine class probabilities have been statistically integrated.")
    return coarse_prob, fine_prob, predicted_item

def find_suitable_foods(association_rule_csv, coarse_prob, predicted_items):
    # Save df, easy to merge with csv to calculate final score.
    global confidence_dishes_df  

    # Remove duplicate input item
    predicted_items = list(dict.fromkeys(predicted_items))
    print("User food input de-duplication")

    coarse_breast = coarse_prob["coarse_breast"]
    coarse_sandwich = coarse_prob["coarse_sandwich"]
    coarse_wrap = coarse_prob["coarse_wrap"]

    # coarse_prob_value = coarse_prob
    coarse_prob_sum = 0
    for i in range(len(predicted_items)):
        coarse_prob_sum += coarse_breast[i] + coarse_sandwich[i] + coarse_wrap[i]

    # coarse_prob_value = coarse_prob_sum / (len(predicted_item)*3)
    coarse_prob_value = np.median(coarse_prob_sum)

    if not isinstance(predicted_items, list):
        predicted_items = [predicted_items]

    # From CSV load rules
    rules_df = pd.read_csv(association_rule_csv) 

    rules_df['Fine_Class_Consequent'] = rules_df['Fine_Class_Consequent'].astype(str)

    # From the rules, find the food groups that contain the input food type
    suitable_rules = rules_df[rules_df['Fine_Class_Consequent'].apply(lambda x: any(food in x for food in predicted_items))]
    # print(suitable_rules.sort_values(by='Fine_Class_Confidence', ascending=False))

    print("Association rules related to user input have been identified.")

    # Calculate final confidence for each suitable rule
    suitable_rules['Final_Confidence'] = suitable_rules.apply(lambda row: coarse_prob_value * row['Coarse_Class_Confidence'] + (1 - coarse_prob_value) * row['Fine_Class_Confidence'], axis=1)

    # Save food group and confidence score for suitable_rules
    suitable_foods = []

    for index, row in suitable_rules.iterrows():
        dish = row['Fine_Class_Antecedent']
        confidence = row['Final_Confidence']

        # Extract individual food items from the dish string
        foods = [food.strip() for food in dish.split(',')]

        # Add each individual food item and the combination with the input food items
        for food in foods:
            suitable_foods.append((food, confidence))
            for input_food in predicted_items:
                combination = f'{input_food},{food}'
                suitable_foods.append((combination, confidence))
    print("Association rule has been drawn up for recommended combinations")
    # Create DataFrame and sort by AR_Score
    confidence_dishes_df = pd.DataFrame(suitable_foods, columns=['Dish', 'AR_Score'])
    confidence_dishes_df = confidence_dishes_df.sort_values(by='AR_Score', ascending=False)
    return confidence_dishes_df

def calculate_nutrition_gap(coarse_prob, fine_prob, nutrition_dataset):
    nutrition_df = pd.read_csv(nutrition_dataset)

    #print(coarse_prob)
    coarse_breast = coarse_prob["coarse_breast"]
    coarse_sandwich = coarse_prob["coarse_sandwich"]
    coarse_wrap = coarse_prob["coarse_wrap"]

    basil_garlic_chicken_breast = fine_prob["basil_garlic_chicken_breast"]
    basil_garlic_chicken_breast_2 = fine_prob["basil_garlic_chicken_breast_2"]
    black_pepper_chicken_breast = fine_prob["black_pepper_chicken_breast"]
    herbs_chicken_breast = fine_prob["herbs_chicken_breast"]
    bacon_egg_mayo_sandwich = fine_prob["bacon_egg_mayo_sandwich"]
    ham_cheese_pickle_sandwich = fine_prob["ham_cheese_pickle_sandwich"]
    smoked_salmon_egg_mayo_sandwich = fine_prob["smoked_salmon_egg_mayo_sandwich"]
    super_club_sandwich = fine_prob["super_club_sandwich"]
    caesar_chicken_cold_wrap = fine_prob["caesar_chicken_cold_wrap"]
    peking_duck_cold_wrap = fine_prob["peking_duck_cold_wrap"]
    sesame_chicken_with_baby_spinach_cold_wrap = fine_prob["sesame_chicken_with_baby_spinach_cold_wrap"]
    smoked_salmon_egg_salad_cold_wrap = fine_prob["smoked_salmon_egg_salad_cold_wrap"]

    nutritional_values = {
        "Energy": [],
        "Protein": [],
        "TotalFat": [],
        "SaturatedFat": [],
        "TransFat": [],
        "Carbohydrate": [],
        "Sugars": [],
        "Sodium": []
    }

    daliy_nutrition_need = [960, 15, 21.6, 96, 9.6, 132, 96, 800]

    for i in range(len(coarse_prob["coarse_breast"])):
        breast_energy = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'Energy'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'Energy'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'Energy'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'Energy'].values[0]
        )
        sandwich_energy = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'Energy'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'Energy'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'Energy'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'Energy'].values[0]
        )

        wrap_energy = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'Energy'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'Energy'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'Energy'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'Energy'].values[0]
        )
        nutritional_values["Energy"].append(breast_energy + sandwich_energy + wrap_energy)

        breast_protein = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'Protein'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'Protein'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'Protein'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'Protein'].values[0]
        )

        sandwich_protein = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'Protein'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'Protein'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'Protein'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'Protein'].values[0]
        )

        wrap_protein = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'Protein'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'Protein'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'Protein'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'Protein'].values[0]
        )
        nutritional_values["Protein"].append(breast_protein + sandwich_protein + wrap_protein)

        breast_TotalFat = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'TotalFat'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'TotalFat'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'TotalFat'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'TotalFat'].values[0]
        )

        sandwich_TotalFat = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'TotalFat'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'TotalFat'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'TotalFat'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'TotalFat'].values[0]
        )

        wrap_TotalFat = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'TotalFat'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'TotalFat'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'TotalFat'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'TotalFat'].values[0]
        )
        nutritional_values["TotalFat"].append(breast_TotalFat + sandwich_TotalFat + wrap_TotalFat)

        breast_SaturatedFat = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'SaturatedFat'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'SaturatedFat'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'SaturatedFat'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'SaturatedFat'].values[0]
        )

        sandwich_SaturatedFat = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'SaturatedFat'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'SaturatedFat'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'SaturatedFat'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'SaturatedFat'].values[0]
        )

        wrap_SaturatedFat = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'SaturatedFat'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'SaturatedFat'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'SaturatedFat'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'SaturatedFat'].values[0]
        )
        nutritional_values["SaturatedFat"].append(breast_SaturatedFat + sandwich_SaturatedFat + wrap_SaturatedFat)

        breast_TransFat = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'TransFat'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'TransFat'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'TransFat'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'TransFat'].values[0]
        )

        sandwich_TransFat = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'TransFat'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'TransFat'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'TransFat'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'TransFat'].values[0]
        )

        wrap_TransFat = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'TransFat'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'TransFat'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'TransFat'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'TransFat'].values[0]
        )
        nutritional_values["TransFat"].append(breast_TransFat + sandwich_TransFat + wrap_TransFat)

        breast_Carbohydrate = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'Carbohydrate'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'Carbohydrate'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'Carbohydrate'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'Carbohydrate'].values[0]
        )

        sandwich_Carbohydrate = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'Carbohydrate'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'Carbohydrate'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'Carbohydrate'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'Carbohydrate'].values[0]
        )

        wrap_Carbohydrate = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'Carbohydrate'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'Carbohydrate'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'Carbohydrate'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'Carbohydrate'].values[0]
        )
        nutritional_values["Carbohydrate"].append(breast_Carbohydrate + sandwich_Carbohydrate + wrap_Carbohydrate)

        breast_Sugars = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'Sugars'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'Sugars'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'Sugars'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'Sugars'].values[0]
        )

        sandwich_Sugars = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'Sugars'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'Sugars'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'Sugars'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'Sugars'].values[0]
        )

        wrap_Sugars = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'Sugars'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'Sugars'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'Sugars'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'Sugars'].values[0]
        )
        nutritional_values["Sugars"].append(breast_Sugars + sandwich_Sugars + wrap_Sugars)

        breast_Sodium = coarse_breast[i] * (
            basil_garlic_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Basil_&_Garlic_Chicken_Breast', 'Sodium'].values[0] +
            basil_garlic_chicken_breast_2[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sweet_Pepper_Garlic_Chicken_Breast', 'Sodium'].values[0] +
            black_pepper_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Black_Pepper_Chicken_Breast', 'Sodium'].values[0] +
            herbs_chicken_breast[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Herbs_Chicken_Breast', 'Sodium'].values[0]
        )

        sandwich_Sodium = coarse_sandwich[i] * (
            bacon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Bacon_&_Egg_Mayo_Sandwich', 'Sodium'].values[0] +
            ham_cheese_pickle_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Ham_Cheese_&_Pickle_Sandwich', 'Sodium'].values[0] +
            smoked_salmon_egg_mayo_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Mayo_Sandwich', 'Sodium'].values[0] +
            super_club_sandwich[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Super_Club_Sandwich', 'Sodium'].values[0]
        )

        wrap_Sodium = coarse_wrap[i] * (
            caesar_chicken_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Caesar_Chicken_Cold_Wrap', 'Sodium'].values[0] +
            peking_duck_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Peking_Duck_Cold_Wrap', 'Sodium'].values[0] +
            sesame_chicken_with_baby_spinach_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap', 'Sodium'].values[0] +
            smoked_salmon_egg_salad_cold_wrap[i] * nutrition_df.loc[nutrition_df['ProductName'] == 'Smoked_Salmon_&_Egg_Salad_Cold_Wrap', 'Sodium'].values[0]
        )
        nutritional_values["Sodium"].append(breast_Sodium + sandwich_Sodium + wrap_Sodium)
    #print(nutritional_values)
    print("It has been concluded that there is approximately how much nutrition is currently available")

    columns = ["Energy","Protein","TotalFat","SaturatedFat","TransFat","Carbohydrate","Sugars","Sodium"]
    # print(len(nutritional_values))
    # print(nutritional_values["Sodium"][0])
    nutrition_gap = []
    for i in range(len(nutritional_values)):
        temp = columns[i]
        nutrition_gap.append(daliy_nutrition_need[i]-nutritional_values[temp][0])

    print("Nutritional gap of the current user meal has been identified.")
    return nutrition_gap

#def recommend_dish(sum_Energy,sum_Protein,sum_TotalFat,sum_SaturatedFat,sum_TransFat,sum_Carbohydrate,sum_Sugars,sum_Sodium):
def recommend_dish(preprocess_csv, nutrition_array):
    df = pd.read_csv(preprocess_csv)

    df['sum_Energy'] = df['sum_Energy'].fillna(0)
    df['sum_Protein'] = df['sum_Protein'].fillna(0)
    df['sum_TotalFat'] = df['sum_TotalFat'].fillna(0)
    df['sum_SaturatedFat'] = df['sum_SaturatedFat'].fillna(0)
    df['sum_TransFat'] = df['sum_TransFat'].fillna(0)
    df['sum_Carbohydrate'] = df['sum_Carbohydrate'].fillna(0)
    df['sum_Sugars'] = df['sum_Sugars'].fillna(0)
    df['sum_Sodium'] = df['sum_Sodium'].fillna(0)

    # From the preporcess df, take the nutrition data to facilitate the calculation of cosine similarity.
    features = df[['sum_Energy','sum_Protein','sum_TotalFat','sum_SaturatedFat','sum_TransFat','sum_Carbohydrate','sum_Sugars','sum_Sodium']]

    # standardation
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Store Input
    #input_nutrition = [[sum_Energy,sum_Protein,sum_TotalFat,sum_SaturatedFat,sum_TransFat,sum_Carbohydrate,sum_Sugars,sum_Sodium]]
    input_nutrition = [nutrition_array]

    # 將input_nutrition standardation
    scaled_input = scaler.transform(input_nutrition)

    # standardation後的features造KD Tree
    tree = KDTree(scaled_features)

    print("Nutrition gap has been imported to KD-Tree")

    # Tree search to find nearest neighbours
    n_neighbors = round(df.shape[0]/3) # 查詢最近鄰居(whole tree 33.333%)
    dist, indices = tree.query(scaled_input, k=n_neighbors)

    # Based on the found neighbouring data, calculate their cosine similarity score.
    tree_features = df[['sum_Energy','sum_Protein','sum_TotalFat','sum_SaturatedFat','sum_TransFat','sum_Carbohydrate','sum_Sugars','sum_Sodium']].iloc[indices[0]]
    scaled_tree_features = scaler.transform(tree_features)
    sim_scores = cosine_similarity(scaled_input, scaled_tree_features)
    sim_scores = sim_scores.flatten()

    print("Cosine similarity has been derived.")

    # Create DF
    recommend_df = pd.DataFrame({
        'Dish': df['Dishes'].iloc[indices[0]],
        'Type': df['Types'].iloc[indices[0]],
        'sum_Energy': df['sum_Energy'].iloc[indices[0]],
        'sum_Protein': df['sum_Protein'].iloc[indices[0]],
        'sum_TotalFat': df['sum_TotalFat'].iloc[indices[0]],
        'sum_SaturatedFat': df['sum_SaturatedFat'].iloc[indices[0]],
        'sum_TransFat': df['sum_TransFat'].iloc[indices[0]],
        'sum_Carbohydrate': df['sum_Carbohydrate'].iloc[indices[0]],
        'sum_Sugars': df['sum_Sugars'].iloc[indices[0]],
        'sum_Sodium': df['sum_Sodium'].iloc[indices[0]],
        'CS_Score': sim_scores,
        'Distance': dist[0]
    })

    # Sort by distance (the smaller the better) and return the top-ranked recommendations
    recommend_df = recommend_df.sort_values(by='Distance')

    return recommend_df.reset_index(drop=True)

import pandas as pd

def combined_score(nutrition_gap, coarse_prob, food_type_input, association_rule_csv, preprocess_csv):
    Similarity_Weight = 101.0
    Association_Weight = 1.0

    # Derive confidence_dishes_df by processing input with function
    confidence_dishes_df = find_suitable_foods(coarse_prob, food_type_input, association_rule_csv)
    # Use function to handle inputs to derive recommend_df.
    recommend_df = recommend_dish(preprocess_csv, nutrition_gap)

    # Merger of Csv under ‘Type’
    merged_df = recommend_df.merge(confidence_dishes_df, on='Dish', how='left')

    merged_df['AR_Score'] = merged_df['AR_Score'].fillna(0)

    #return merged_df
    # Calculate final score
    combined_df = merged_df[['Dish', 'Type', 'CS_Score', 'AR_Score']]  # 拿需要的columns

    #return merged_df
    combined_df['Final_Score'] = (combined_df['CS_Score'] * Similarity_Weight) + (combined_df['AR_Score'] * Association_Weight)

    # Delete rows containing NaN values in ‘Combined_Score’
    combined_df = combined_df.dropna()

    # Sort by ‘Combined_Score’ descending order
    sorted_combined_df = combined_df.sort_values(by='Final_Score', ascending=False)
    # print(sorted_combined_df)

    return sorted_combined_df
# -*- coding: utf-8 -*-
import pandas as pd
from itertools import combinations

def generate_comb_of_foods():
    df = pd.read_csv('7select_Product.csv')
    max_combine = 5
    # Generate combinations in groups of 1-5
    all_combinations = []
    for r in range(1, max_combine+1):
        for combo in combinations(df.iterrows(), r):
            dishes = [item[1] for item in combo]
            dishes_name = ','.join(set(dish['ProductName'] for dish in dishes))
            sum_Energy = sum(dish['Energy'] for dish in dishes)
            sum_Protein = sum(dish['Protein'] for dish in dishes)
            sum_TotalFat = sum(dish['TotalFat'] for dish in dishes)
            sum_SaturatedFat = sum(dish['SaturatedFat'] for dish in dishes)
            sum_TransFat = sum(dish['TransFat'] for dish in dishes)
            sum_Carbohydrate = sum(dish['Carbohydrate'] for dish in dishes)
            sum_Sugars = sum(dish['Sugars'] for dish in dishes)
            sum_Sodium = sum(dish['Sodium'] for dish in dishes)
            types = ','.join(sorted(set(dish['Type'] for dish in dishes)))
            all_combinations.append({
                #'Dishes': [dish['ProductName'] for dish in dishes],
                'Dishes': dishes_name,
                'sum_Energy': sum_Energy,
                'sum_Protein': sum_Protein,
                'sum_TotalFat': sum_TotalFat,
                'sum_SaturatedFat': sum_SaturatedFat,
                'sum_TransFat': sum_TransFat,
                'sum_Carbohydrate': sum_Carbohydrate,
                'sum_Sugars': sum_Sugars,
                'sum_Sodium': sum_Sodium,
                'Types': types
            })
    
    combo_df = pd.DataFrame(all_combinations)
    
    combo_df.to_csv("7select_Product_preprocess.csv", sep=',', encoding='utf-8')
    
    print("Preprocessed")

if __name__ == "__main__":
    generate_comb_of_foods()

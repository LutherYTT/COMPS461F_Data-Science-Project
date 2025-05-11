# -*- coding: utf-8 -*-
import numpy as np
import csv

P1 = 0.8  # Probability that food categories in the same combination do not repeat
P2 = 0.85  # Probability that the combination includes a Meat/Seafood + Vegetable (Fruit_Cup/Salad/Vegetable) pairing
P3 = 0.85  # Probability that the combination contains no more than one staple food (Rice/Noodle)
P4 = 0.7   # Probability that the combination contains no more than one drink (Drink)
P5 = 0.75  # Probability that Sandwich and Wrap do not appear in the same combination
# P6 = 0.75  # Probability that Meat and Seafood do not appear in the same combination (commented out)
noise_ratio = 0.1  # Noise ratio
generate_count = 1000  # Number of combinations to generate

# Food categories
categories = ['Drink', 'Meat', 'Seafood', 'Rice', 'Noodle',
              'Salad', 'Fruit_cup', 'Vegetable', 'Sandwich', 'Wrap']

product_names = {
    'Meat': [
        'Herbs_Chicken_Breast', 'Basil_&_Garlic_Chicken_Breast',
        'Black_Pepper_Chicken_Breast', 'Sweet_Pepper_Garlic_Chicken_Breast'
    ],
    'Sandwich': [
        'Ham_Cheese_&_Pickle_Sandwich',
        'Super_Club_Sandwich',
        'Smoked_Salmon_&_Egg_Mayo_Sandwich',
        'Bacon_&_Egg_Mayo_Sandwich'
    ],
    'Salad': [
        'French_Style_Tuna_&_Egg_Salad',
        'Chef_Salad',
        'Chicken_Caesar_Salad',
    ],
    'Fruit_cup': [
        'Cut-Fruit_Cup_Healthy_Mix',
        'Cut-Fruit_Cup_Seasonal_Mix'
    ],
    'Wrap': [
        'Peking_Duck_Cold_Wrap',
        'Sesame_Chicken_with_Baby_Spinach_Cold_Wrap',
        'Smoked_Salmon_&_Egg_Salad_Cold_Wrap',
        'Caesar_Chicken_Cold_Wrap'
    ],
    # Real products
    'Rice': [
        'Yeung_Chow_Fried_Rice',
        'Steamed_Rice_with_Pork_Ribs_&_Chicken_Paw',
        'Steamed_Rice_with_Dried_Squid_&_Meat_Patty'
    ],
    'Noodle': [
        'Spaghetti_with_Meat_Sauce',
        'Singapore_Style_Fried_Rice_Vermicelli'
    ],
    'Vegetable': [
        'Grilled_Vegetables',
        'Mixed_Veggie_Stir_Fry'
    ],
    'Drink': [
        'Iced_Tea',
        'Lemonade',
        'Lemon_Tea'
    ],
    'Seafood': [
        'Grilled_Salmon',
        'Shrimp_Scampi'
    ]
}

def skewed_choice(options, skewness=2):
    probabilities = np.arange(len(options), 0, -1, dtype=float) ** skewness
    probabilities /= probabilities.sum()
    return np.random.choice(options, p=probabilities)

# Generate food combination
def generate_combination():
    # Decide whether to generate noise data
    # if np.random.rand() < noise_ratio:
    #     # Randomly select 2-8 categories with skewed distribution
    #     n = skewed_choice(np.arange(2, 9))
    #     # Generate completely random combination (No rules)
    #     return ', '.join(np.random.choice(categories, n, replace=True))

    # Randomly select 2-8 categories with skewed distribution
    n = skewed_choice(np.arange(2, 9))
    selected = []

    # Rule 2: With probability P2, combination includes Meat/Seafood + Vegetable (Fruit_Cup/Salad/Vegetable)
    if np.random.rand() < P2:
        meat_seafood = np.random.choice(['Meat', 'Seafood'])
        vegetable = np.random.choice(['Salad', 'Fruit_cup', 'Vegetable'])
        selected.extend([meat_seafood, vegetable])
        remaining = n - 2
    else:
        remaining = n

    # If there are remaining slots, add other food categories
    for _ in range(remaining):
        available = categories.copy()

        # Rule 1: With probability P1, no repeated food categories in the same combination
        if np.random.rand() < P1:
            available = [c for c in available if c not in selected]

        # Rule 3: With probability P3, no more than one staple food (Rice/Noodle) in the combination
        if np.random.rand() < P3:
            staple_count = sum(1 for c in selected if c in ['Rice', 'Noodle'])
            if staple_count >= 1:
                available = [c for c in available if c not in ['Rice', 'Noodle']]

        # Rule 4: With probability P4, no more than one drink (Drink) in the combination
        if np.random.rand() < P4:
            drink_count = sum(1 for c in selected if c == 'Drink')
            if drink_count >= 1:
                available = [c for c in available if c != 'Drink']

        # Rule 5: With probability P5, Sandwich and Wrap do not appear in the same combination
        if np.random.rand() < P5:
            has_sandwich = 'Sandwich' in selected
            has_wrap = 'Wrap' in selected
            if has_sandwich:
                available = [c for c in available if c != 'Wrap']
            elif has_wrap:
                available = [c for c in available if c != 'Sandwich']

        # # Rule 6: With probability P6, Meat and Seafood do not appear in the same combination (commented out)
        # if np.random.rand() < P6:
        #     has_meat = 'Meat' in selected
        #     has_seafood = 'Seafood' in selected
        #     if has_meat:
        #         available = [c for c in available if c != 'Seafood']
        #     elif has_seafood:
        #         available = [c for c in available if c != 'Meat']

        # If no suitable food category is available, allow repeats
        if not available:
            available = categories.copy()

        next_item = np.random.choice(available)
        selected.append(next_item)

    return ', '.join(selected[:n])

# Assign product names randomly
def assign_product_names(selected_categories):
    product_name_options = {}

    for category in selected_categories:
        if category in product_names:
            if category not in product_name_options:
                product_name_options[category] = product_names[category]

    chosen_products = []
    for category in selected_categories:
        if category in product_name_options:
            # Randomly choose one product name
            chosen_product = np.random.choice(product_name_options[category])
            chosen_products.append(chosen_product)

    return ', '.join(chosen_products)

# Generate CSV file
def save_to_csv(filename, num_samples):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['items', 'product_name'])

        for _ in range(num_samples):
            combination = generate_combination()
            selected_categories = combination.split(', ')
            product_names_str = assign_product_names(selected_categories)
            writer.writerow([combination, product_names_str])

if __name__ == "__main__":
    np.random.seed(42)  # Fix random seed for reproducibility
    save_to_csv('fine_food_combinations.csv', generate_count)
    
    # For monitoring output
    print("items, product_name")
    for _ in range(generate_count):
        combination = generate_combination()
        selected_categories = combination.split(', ')
    product_name = assign_product_names(selected_categories)
    print(f'"{combination}", "{product_name}"')

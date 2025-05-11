# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageDraw
from concurrent.futures import ThreadPoolExecutor

# set random seed for easy recovery
random_seed = 42
random.seed(random_seed)

# File Location
background_folder = './background/'
food_folder = './food/'
misc_folder = './misc/'
output_folder = './output/images/'
labels_fine_folder = './output/labels_fine/'
labels_coarse_folder = './output/labels_coarse/'

# parameter
min_scale = 0.7 # Minimum random size, 0.7 = 70%
max_scale = 2.0 # Maximum random size, 2.0 = 200%
max_rotation = 180  # Maximum random rotation angle
max_blur = 3 # Maximum blur of the random, eg. 2 = 50%
min_brightness = 0.3 # Minimum brightness of the random, 0.3 = 30%
max_brightness = 1.4 # Random max_brightness, 1.4 = 140%
max_food_item = 5 # Maximum number of food items to add.
max_misc_item = 3 # Maximum number of misc items to be added.
smooth_factor = 0.4 # Smooth the scale factor of random_sclae function, to avoid pure random to later scale is basically equal to max_scale.
min_contrast = 0.6 # Random Minimum Contrast
max_contrast = 1.8 # Random Maximum Contrast

# Confirm folder exists
os.makedirs(output_folder, exist_ok=True)
os.makedirs(labels_fine_folder, exist_ok=True)
os.makedirs(labels_coarse_folder, exist_ok=True)

# class mapping
# These all correspond to folder names.
# fine_class_mappings = {
#     "cut_fruit_cup_healthy_mix": 0,
#     "cut_fruit_cup_seasonal_mix": 1,
#     "basil_garlic_chicken_breast": 2,
#     "basil_garlic_chicken_breast(2)": 3,
#     "black_pepper_chicken_breast": 4,
#     "herbs_chicken_breast": 5,
#     "assorted_salad": 6,
#     "chef_salad": 7,
#     "chicken_caesar_salad": 8,
#     "french_style_tuna_egg_salad": 9,
#     "bacon_egg_mayo_sandwich": 10,
#     "ham_cheese_pickle_sandwich": 11,
#     "smoked_salmon_egg_mayo_sandwich": 12,
#     "super_club_sandwich": 13,
#     "caesar_chicken_cold_wrap": 14,
#     "peking_duck_cold_wrap": 15,
#     "sesame_chicken_with_baby_spinach_cold_wrap": 16,
#     "smoked_salmon_egg_salad_cold_wrap": 17,
#     "misc": 18
# }

# These all correspond to folder names.
fine_class_mappings = {
    "basil_garlic_chicken_breast": 0,
    "basil_garlic_chicken_breast(2)": 1,
    "black_pepper_chicken_breast": 2,
    "herbs_chicken_breast": 3,
    "bacon_egg_mayo_sandwich": 4,
    "ham_cheese_pickle_sandwich": 5,
    "smoked_salmon_egg_mayo_sandwich": 6,
    "super_club_sandwich": 7,
    "caesar_chicken_cold_wrap": 8,
    "peking_duck_cold_wrap": 9,
    "sesame_chicken_with_baby_spinach_cold_wrap": 10,
    "smoked_salmon_egg_salad_cold_wrap": 11,
    "misc": 12
}

# These all correspond to folder names.
# coarse_class_mappings = {
#     "fruit_cup": 0,
#     "meat": 1,
#     "salad": 2,
#     "sandwich": 3,
#     "wrap": 4,
#     "misc": 5
# }

coarse_class_mappings = {
    "meat": 0,
    "sandwich": 1,
    "wrap": 2,
    "misc": 3
}

# Randomly selects a picture from a specified folder.
def random_image(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if files:
        return os.path.join(folder, random.choice(files))
    return None

# Where will the random picture be on the background?
def random_position(background_width, background_height, image_width, image_height):
    # Ensure that the food/misc picture is not too close to the background edge.
    min_x = int(0.04 * background_width)
    min_y = int(0.04 * background_height)
    max_x = background_width - image_width - min_x
    max_y = background_height - image_height - min_y

    x = round(random.randint(min_x, max_x) + (random.randint(min_x, max_x)/10)) if max_x > min_x else min_x
    y = round(random.randint(min_y, max_y) + (random.randint(min_y, max_y)/10)) if max_y > min_y else min_y
    return x, y

# Randomly scale food/misc from min_scale to max_scale.
def random_scale(image, min_scale_factor, max_scale_factor):
    # Because the earlier the generation, the more at the bottom,
    # so control the min scale, so that the min scale can not be lower than the prev_scale in order to achieve the effect of near big and far small.

    #scale_factor = random.uniform(min_scale_factor, max_scale)
    scale_factor = random.uniform(min_scale_factor, ((1 - smooth_factor) * min_scale_factor) + (smooth_factor * max_scale_factor))
    #scale_factor = ((1 - smooth_factor) * min_scale_factor) + (smooth_factor * scale_factor)

    #print(f"prev_scale: {min_scale_factor}")
    #print(f"scale_factor: {scale_factor}")

    # Changing the scale of a picture
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return scale_factor, image.resize((new_width, new_height), Image.LANCZOS)

# Because the difference in resolution between the background and food/misc is too big, avoiding food/misc is only a small piece in the background.
# Ensure that food/misc is not smaller than the minimum size.
def ensure_min_size(image, min_width, min_height):
    width, height = image.size
    if width < min_width or height < min_height:
    # When the width or height of the food/misc picture is lower than the lower limit,
    # calculate how many times lower it is, and use this multiplier to scale the food/misc picture.
        scale_x = min_width / width if width < min_width else 1
        scale_y = min_height / height if height < min_height else 1
        # Make the shortest edge equal to min_width/min_height.
        scale = max(scale_x, scale_y)
        # Use this ratio to zoom in and out of food/misc pictures.
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

# Because of the difference in aspect ratio between the images,
# it is difficult to make the effect of near big and far small, so we directly re-size all the images.
def force_orginal_size_scale(image, min_width, min_height):
    width, height = image.size
    scale_x = min_width / width
    scale_y = min_height / height
    # Make the longest side equal to min_width/min_height.
    scale = min(scale_x, scale_y)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.LANCZOS)

def force_orginal_size(image, min_width, min_height):
    return image.resize((min_width, min_height), Image.LANCZOS)

# Randomly rotating food/misc pictures
def random_rotation(image):
    angle = random.randint(0, max_rotation)
    rotated_image = image.rotate(angle, expand=True, resample=Image.BILINEAR)
    return rotated_image

# Random Blur
def random_blur(image):
    blur_amount = random.uniform(0, max_blur)
    return image.filter(ImageFilter.GaussianBlur(radius=blur_amount))

# Random brightness of the overall picture
def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    brightness_factor = random.uniform(min_brightness, max_brightness)  # Adjust brightness between 80% and 120%
    return enhancer.enhance(brightness_factor)

# Random Picture Flip
def random_flip(image):
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

# Random Contrast Adjustment
def random_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    contrast_factor = random.uniform(min_contrast, max_contrast)
    return enhancer.enhance(contrast_factor)

# Because the difference in resolution between the background and the food/misc image is too large,
# avoiding the food/misc image is only a small piece in the background.
# Adjust the background size to avoid the background being too big
def resize_background(background, max_width, max_height):
    width, height = background.size
    if width > max_width or height > max_height:
    # When the width or height of the background exceeds the set limit,
    # calculate how many times it exceeds the set limit,
    # and use this multiplier to scale the background.
        # How many multiples of the calculation have been exceeded?
        scale_x = max_width / width if width > max_width else 1
        scale_y = max_height / height if height > max_height else 1
        scale = min(scale_x, scale_y)
        # Use this ratio to zoom in on the background.
        new_width = int(width * scale)
        new_height = int(height * scale)
        return background.resize((new_width, new_height), Image.LANCZOS)
    return background

# Cut out the blank spaces for re-trimming the picture after rotation.
def trim_image(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg) # Increased contrast for better detection
    diff = ImageChops.add(diff, diff, 5.0)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    else:
        return image  # If there are no significant differences, the original image is returned.

# Combine background, food and misc to create data.
def compose_image(image_id):
    # When start generating a image, reset prev_scale.
    # Used to store the last scale factor
    prev_scale = min_scale
    #print(prev_scale)
    # 1. random select background
    background_path = random_image(background_folder)
    background = Image.open(background_path).convert("RGB")
    #background = resize_background(background, 1920, 1080)  # resize background image size
    background = force_orginal_size(background, 832, 832)
    background_width, background_height = background.size

    # for saving label
    fine_labels = []
    coarse_labels = []

    # 2. Add 1 to max_food_item pieces of food
    num_food_items = random.randint(1, max_food_item)
    for _ in range(num_food_items):
        # random select food type folder
        food_category = random.choice([d for d in os.listdir(food_folder) if os.path.isdir(os.path.join(food_folder, d))])
        food_category_path = os.path.join(food_folder, food_category)

        # random select fine food
        food_subcategory = random.choice([d for d in os.listdir(food_category_path) if os.path.isdir(os.path.join(food_category_path, d))])
        food_subcategory_path = os.path.join(food_category_path, food_subcategory)
        # In the fine food folder, select a picture at random.
        food_image_path = random_image(food_subcategory_path)
        if food_image_path:
            try:
                food_image = Image.open(food_image_path).convert("RGBA")
                # random transform
                #food_image = ensure_min_size(food_image, 200, 200)
                food_image = force_orginal_size_scale(food_image, 180, 180)
                prev_scale ,food_image = random_scale(food_image, prev_scale, max_scale) # 隨機縮放
                #print(food_image.size)
                food_image = random_rotation(food_image) # random rotation
                food_image = random_blur(food_image) # random blur
                food_image = random_flip(food_image) # random flip

                # triming blank spaces
                food_image = trim_image(food_image)

                # Get the length/width ratio of the image
                food_width, food_height = food_image.size
                x, y = random_position(background_width, background_height, food_width, food_height)  # 會隨機放置在背景上的什麼位置?

                # Place food_image on the background.
                background.paste(food_image, (x, y), food_image)

                # save fine label
                fine_class_id = fine_class_mappings.get(food_subcategory, -1)
                if fine_class_id != -1:
                    x_center = (x + food_width / 2) / background_width
                    y_center = (y + food_height / 2) / background_height
                    width = food_width / background_width
                    height = food_height / background_height
                    fine_labels.append(f"{fine_class_id} {x_center} {y_center} {width} {height}")

                # save coarse label
                coarse_class_id = coarse_class_mappings.get(food_category, -1)
                if coarse_class_id != -1:
                    coarse_labels.append(f"{coarse_class_id} {x_center} {y_center} {width} {height}")

            except Exception as e:
                print(f"Error processing food image: {e}")

    # 3. Add up to max_misc_item individual misc
    num_misc_items = random.randint(0, max_misc_item)
    for _ in range(num_misc_items):
        misc_image_path = random_image(misc_folder)
        if misc_image_path:
            try:
                # Random Transform
                misc_image = Image.open(misc_image_path).convert("RGBA")
                #misc_image = ensure_min_size(misc_image, 100, 100)
                misc_image = force_orginal_size_scale(misc_image, 240, 240)

                prev_scale, misc_image = random_scale(misc_image, prev_scale, max_scale)
                #print(misc_image.size)
                misc_image = random_rotation(misc_image)
                misc_image = random_blur(misc_image)
                misc_image = random_flip(misc_image)

                # triming blank spaces
                misc_image = trim_image(misc_image)

                # Get the length/width ratio of the image
                misc_width, misc_height = misc_image.size
                x, y = random_position(background_width, background_height, misc_width, misc_height)

                # Placing misc_image on the background
                background.paste(misc_image, (x, y), misc_image)

                # Save label, label as misc
                x_center = (x + misc_width / 2) / background_width
                y_center = (y + misc_height / 2) / background_height
                width = misc_width / background_width
                height = misc_height / background_height
                fine_labels.append(f"{fine_class_mappings['misc']} {x_center} {y_center} {width} {height}")
                coarse_labels.append(f"{coarse_class_mappings['misc']} {x_center} {y_center} {width} {height}")

            except Exception as e:
                print(f"Error processing misc image: {e}")

    # Overall Random Brightness and Contrast of Images
    background = random_brightness(background)
    background = random_contrast(background)

    # Save synthetic image
    output_image_path = os.path.join(output_folder, f"synthetic_{image_id}.png")
    background.save(output_image_path)

    # save the label
    with open(os.path.join(labels_fine_folder, f"synthetic_{image_id}.txt"), 'w') as f:
        for label in fine_labels:
            f.write(label + '\n')

    with open(os.path.join(labels_coarse_folder, f"synthetic_{image_id}.txt"), 'w') as f:
        for label in coarse_labels:
            f.write(label + '\n')

# Multi-threaded to speed up image generation
def generate_images(num_images):
    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(compose_image, range(num_images))
        
if __name__ == "__main__":
    num_images = 5000
    generate_images(num_images)
    print(f"Generated {num_images} images with multi-threading.")

    # for i in range(num_images):
        # compose_image(i)
        # print(f"Generated {i} images.")

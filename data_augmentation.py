import os
import pandas as pd
import Augmentor
import shutil
from PIL import Image
import random
    # Define the label mapping
label_mapping = {
        '10X_FOV 100 Anabaena WILL CRASH': 'Straight Anabaena',
        'Example_Anabaena-coiled_10X_TR': 'Coiled Anabaena',
        'Example_Anabaena-straight_10X_TR': 'Straight Anabaena',
        'Anabaena Curved 10X FOV 100': 'Coiled Anabaena',
        'Anabaena 10X FOV 100': 'Straight Anabaena',
        'Example_Aphanazomenon_10X_TR': 'Aphanazomenon',
        'Example_Asterionella_10X_TR': 'Asterionella',
        'Asterionella 10X FOV100': 'Asterionella',
        'Example_Cyclotella_10X_TR': 'Cyclotella',
        'Cyclotella 10X FOV100': 'Cyclotella',
        'Example_Cylindrospermopsis_10X_TR': 'Cylindrospermopsis',
        'Example_Tabellaria_10X_TR': 'Tabellaria',
        'Example_Lyngbya_10X_TR': 'Lyngbya',
        'Example_Planktothrix_10X_TR': 'Planktothrix',
        'Example_Planktothrix-2_10X_TR': 'Planktothrix',
        'Example_Synedra_10X_TR': 'Synedra',
        'Synedra 10X FOV100': 'Synedra',
        'Synedra': 'Synedra',
        'Fragillaria 10X FOV100': 'Fragillaria'
    }

    # Define the numerical mapping
numerical_mapping = {
        'Straight Anabaena': 1,
        'Coiled Anabaena': 2,
        'Aphanazomenon': 3,
        'Asterionella': 4,
        'Cyclotella': 5,
        'Cylindrospermopsis': 6,
        'Tabellaria': 7,
        'Lyngbya': 8,
        'Planktothrix': 9,
        'Synedra': 10,
        'Fragillaria': 11,
        'Black Hole': 12  # Add the "Black Hole" class
    }
def pad_and_resize_image(image_path, output_path, size=(224, 224)):
    try:
        image = Image.open(image_path)
        
        # Calculate the padding dimensions to make the image square
        width, height = image.size
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2
        
        # Determine the background color by analyzing the border pixels
        border_pixels = (
            list(image.getpixel((0, y)) for y in range(height)) +
            list(image.getpixel((width - 1, y)) for y in range(height)) +
            list(image.getpixel((x, 0)) for x in range(width)) +
            list(image.getpixel((x, height - 1)) for x in range(width))
        )
        background_color = max(set(border_pixels), key=border_pixels.count)
        
        # Create a new square image with the background color
        square_image = Image.new(image.mode, (max_dim, max_dim), background_color)
        square_image.paste(image, (pad_width, pad_height))
        
        # Resize the square image to the desired size
        resized_image = square_image.resize(size)
        
        # Save the resized image
        resized_image.save(output_path)
    except Exception as e:
        print(f"Error occurred while padding and resizing image: {str(e)}")
        
def augment_data(input_dir, sample_size):
    pipeline = Augmentor.Pipeline(input_dir)
    # Define the augmentation operations
    pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom_random(probability=0.5, percentage_area=0.8)
    pipeline.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
    pipeline.flip_top_bottom(probability=0.5)

    try:
        # Generate the augmented samples
        pipeline.sample(sample_size)
    except ValueError as e:
        print(f"Error occurred during augmentation: {str(e)}")
        print("Skipping augmentation for this category.")

def organize_images(csv_file, image_dir, output_dir, black_hole_dir, size=(224, 224)):
    df = pd.read_csv(csv_file, sep=',')
    
    
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        library_id = row['library_id']
        particle = row['particle']
        category = row['name']
        
        # Map the category using the label mapping
        mapped_category = label_mapping.get(category)
        
        if mapped_category:
            # Get the numerical category using the numerical mapping
            numerical_category = numerical_mapping[mapped_category]
            
            image_filename = f"{library_id}_{particle}.jpg"
            new_image_filename = f"{numerical_category}_{particle}.jpg"  # Rename the image
            
            category_dir = os.path.join(output_dir, str(numerical_category))
            os.makedirs(category_dir, exist_ok=True)

            src_path = os.path.join(image_dir, image_filename)
            dst_path = os.path.join(category_dir, new_image_filename)  # Use the new image filename
            pad_and_resize_image(src_path, dst_path, size)
    
    # Process the "Black Hole" class images
    black_hole_category = numerical_mapping['Black Hole']
    black_hole_category_dir = os.path.join(output_dir, str(black_hole_category))
    os.makedirs(black_hole_category_dir, exist_ok=True)

    for image_filename in os.listdir(black_hole_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            particle = os.path.splitext(image_filename)[0]  # Extract the particle ID
            new_image_filename = f"{black_hole_category}_{particle}.jpg"  # Rename the image
            
            src_path = os.path.join(black_hole_dir, image_filename)
            dst_path = os.path.join(black_hole_category_dir, new_image_filename)
            pad_and_resize_image(src_path, dst_path, size)
    
    print("Images organized into category folders.")

def perform_augmentation(train_dir, numerical_mapping, sample_size_per_class=400):
    # Count the number of images per class in the train directory
    class_counts = {}
    for image_filename in os.listdir(train_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            class_num = image_filename.split("_")[0]  # Extract the class number from the image filename
            if class_num not in class_counts:
                class_counts[class_num] = 0
            class_counts[class_num] += 1

    # Perform augmentation for each class
    for class_num, count in class_counts.items():
        sample_size = max(sample_size_per_class - count, 0)
        if sample_size > 0:
            # Create a temporary directory for the class
            temp_dir = os.path.join(train_dir, f"temp_{class_num}")
            os.makedirs(temp_dir, exist_ok=True)

            # Copy the images of the current class to the temporary directory
            for image_filename in os.listdir(train_dir):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    if image_filename.split("_")[0] == class_num:
                        src_path = os.path.join(train_dir, image_filename)
                        dst_path = os.path.join(temp_dir, image_filename)
                        shutil.copy(src_path, dst_path)

            # Perform augmentation on the temporary directory
            augment_data(temp_dir, sample_size)

            # Move the augmented images back to the train directory with unique filenames
            augmented_dir = os.path.join(temp_dir, "output")
            augmented_images = os.listdir(augmented_dir)
            for image in augmented_images:
                if image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    src_path = os.path.join(augmented_dir, image)
                    dst_path = os.path.join(train_dir, f"augmented_{class_num}_{image}")
                    shutil.move(src_path, dst_path)

            # Remove the temporary directory
            shutil.rmtree(temp_dir)

    print("Data augmentation on training set completed.")

def split_data(output_dir, train_test_dir, train_ratio=0.9):
    train_dir = os.path.join(train_test_dir, 'train')
    test_dir = os.path.join(train_test_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for taxa_dir in os.listdir(output_dir):
        taxa_path = os.path.join(output_dir, taxa_dir)
        if os.path.isdir(taxa_path):
            images = [os.path.join(taxa_dir, img) for img in os.listdir(taxa_path)]
            random.shuffle(images)
            
            train_size = int(len(images) * train_ratio)
            train_images = images[:train_size]
            test_images = images[train_size:]
            
            for image in train_images:
                src_path = os.path.join(output_dir, image)
                dst_path = os.path.join(train_dir, os.path.basename(image))
                try:
                    shutil.copy(src_path, dst_path)
                except Exception as e:
                    print(f"Error copying file: {src_path}. Skipping...")
                    print(f"Error message: {str(e)}")
            
            for image in test_images:
                src_path = os.path.join(output_dir, image)
                dst_path = os.path.join(test_dir, os.path.basename(image))
                try:
                    shutil.copy(src_path, dst_path)
                except Exception as e:
                    print(f"Error copying file: {src_path}. Skipping...")
                    print(f"Error message: {str(e)}")
    
    print("Train-test split completed.")
    
# ...

csv_file = "D:/algae_data/target_data.csv"
image_dir = "D:/algae_data/extracted_images"
output_dir = "D:/algae_data/organized_images"
train_test_dir = "D:/algae_data/train_test_data"
black_hole_dir = "D:/algae_data/blackhole"

# organize_images(csv_file, image_dir, output_dir, black_hole_dir, size=(224, 224))

# split_data(output_dir, train_test_dir, train_ratio=0.9)

perform_augmentation(os.path.join(train_test_dir, 'train'), numerical_mapping)
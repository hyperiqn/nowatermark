import os
import random
import shutil

natural_dir = "C:/Users/s_ani/Downloads/27kpng/natural" 
train_images_dir = "C:/Users/s_ani/Downloads/27kpng/train_images/image"  

output_base_dir = "selected_images"
train_natural_output_dir = os.path.join(output_base_dir, "train", "natural")
train_output_dir = os.path.join(output_base_dir, "train", "train")
val_natural_output_dir = os.path.join(output_base_dir, "val", "natural")
val_output_dir = os.path.join(output_base_dir, "val", "train")

os.makedirs(train_natural_output_dir, exist_ok=True)
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_natural_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

natural_files = [f for f in os.listdir(natural_dir) if f.endswith('.jpg')]  
train_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]

matches = []

for natural_file in natural_files:
    natural_file_prefix = os.path.splitext(natural_file)[0] 
    matching_train_images = [train_file for train_file in train_files if train_file.startswith(natural_file_prefix)]
    for train_file in matching_train_images:
        matches.append((natural_file, train_file))

random.shuffle(matches)

train_matches = matches[:4000]
val_matches = matches[4000:5000] 

for natural_file, train_file in train_matches:
    shutil.copy(os.path.join(natural_dir, natural_file), os.path.join(train_natural_output_dir, natural_file))
    shutil.copy(os.path.join(train_images_dir, train_file), os.path.join(train_output_dir, train_file))

for natural_file, train_file in val_matches:
    shutil.copy(os.path.join(natural_dir, natural_file), os.path.join(val_natural_output_dir, natural_file))
    shutil.copy(os.path.join(train_images_dir, train_file), os.path.join(val_output_dir, train_file))

print(f"Copied 4,000 training image pairs and 1,000 validation image pairs to {output_base_dir}")

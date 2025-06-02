import os
from PIL import Image

# Set the root folder where your camera folders are
data_root = 'data/aida_mdp_500'  # change if your path is different
target_size = (1600, 900)  # (width, height)

# List of camera folders
cam_folders = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

for cam in cam_folders:
    cam_dir = os.path.join(data_root, cam)
    if not os.path.exists(cam_dir):
        print(f"Warning: {cam_dir} not found. Skipping.")
        continue

    print(f"Processing images in {cam_dir}...")
    
    for filename in os.listdir(cam_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(cam_dir, filename)

            try:
                with Image.open(img_path) as img:
                    resized_img = img.resize(target_size, Image.BILINEAR)
                    resized_img.save(img_path)  # Overwrite the original
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

print("All images resized to 1600x900.")

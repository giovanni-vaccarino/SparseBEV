import os
import cv2

base_dir = "data/aida_mdp_500"
source_folder = os.path.join(base_dir, "CAM_FRONT")
target_folders = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

black_image = cv2.imread("blackImg.png")

image_names = []
for filename in os.listdir(source_folder):
    lower_name = filename.lower()
    if lower_name.endswith(".png") or lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
        image_names.append(filename)

for name in image_names:
    for folder in target_folders:
        target_path = os.path.join(base_dir, folder, name)
        print(f"Creating image: ${target_path}")
        cv2.imwrite(target_path, black_image)

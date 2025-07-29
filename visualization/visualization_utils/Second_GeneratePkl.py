import json
import pickle
import os
import torch
import numpy as np
from mmdet3d.core.bbox import LiDARInstance3DBoxes # Requires mmdet3d
from scipy.spatial.transform import Rotation as R # Requires scipy

def convert_json_to_structured_pkl(json_filepath, pkl_filepath):
    """
    Reads a JSON file with specific object detection results (like nuScenes format),
    converts relevant data into LiDARInstance3DBoxes and PyTorch tensors,
    and saves it to a .pkl file with the desired structure.

    Args:
        json_filepath (str): The path to the input JSON file (e.g., 'results_nusc.json').
        pkl_filepath (str): The path where the output .pkl file will be saved.
    """
    if not os.path.exists(json_filepath):
        print(f"Error: The JSON file '{json_filepath}' does not exist.")
        return

    try:
        # Load the JSON data
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
        print(f"Successfully loaded data from '{json_filepath}'.")

        # The desired structure is a list of dictionaries, where each dict
        # contains 'boxes_3d', 'scores_3d', and 'labels_3d' for a sample.
        # The 'results' key in your JSON contains data for multiple sample tokens.
        processed_data = []

        # Iterate through each sample in the 'results' dictionary of your JSON
        # The key is the sample_token (e.g., "2025-03-13_11-54_000000")
        for sample_token, detections in json_data['results'].items():
            boxes_list = []
            scores_list = []
            labels_list = []

            for det in detections:
                # Extract translation (x, y, z)
                translation = np.array(det['translation'])

                # Extract size (length, width, height) - assuming this order
                # You might need to verify the exact order expected by LiDARInstance3DBoxes
                # vs. the 'size' field in your JSON (e.g., lwh or wlh)
                size = np.array(det['size'])
                # Reorder size to be (length, width, height) if it's different
                # Example: If your JSON size is (width, length, height) and LiDARInstance3DBoxes expects (length, width, height)
                # size = np.array([size[1], size[0], size[2]])

                # Extract rotation (quaternion: x, y, z, w) and convert to yaw
                # Assuming the quaternion format is [x, y, z, w]
                rotation_quat = np.array(det['rotation'])
                # Create a scipy Rotation object from quaternion
                r = R.from_quat(rotation_quat)
                # Convert to Euler angles (roll, pitch, yaw) in radians
                # 'zyx' is a common convention for autonomous driving (yaw, pitch, roll)
                # We need the yaw angle (rotation around Z-axis)
                euler_angles = r.as_euler('zyx', degrees=False) # or 'zyx' depending on convention
                yaw = euler_angles[1] # Assuming yaw is the third component for 'xyz' or first for 'zyx'
                                     # You may need to adjust this based on your specific coordinate system and mmdet3d's expectation.
                                     # For nuScenes, typically: x: front, y: left, z: up, heading is counter-clockwise around z.
                                     # Check mmdet3d documentation for precise yaw definition (e.g., negative yaw for right turn).

                # Construct the 7-DOF bounding box tensor (x, y, z, dim_x, dim_y, dim_z, yaw)
                # Make sure the order of size components matches mmdet3d's expectation for LiDARInstance3DBoxes
                bbox_tensor_row = np.array([
                    translation[0], translation[1], translation[2],
                    size[0], size[1], size[2],
                    yaw
                ])
                boxes_list.append(bbox_tensor_row)

                # Extract detection score and label
                scores_list.append(det['detection_score'])
                # You'll need a mapping from 'detection_name' to integer label if your labels_3d are integers
                # For this example, assuming 'labels_3d' is directly available or derivable
                # from detection_name or a predefined mapping.
                # If your JSON has integer labels directly, use that. Otherwise, map:
                # e.g., if det['detection_name'] == 'car', map to 0
                # If you have a predefined label mapping, use it here.
                # For now, let's assume `labels_3d` in your provided output implies direct integer labels.
                # If 'labels_3d' in JSON corresponds to your desired output integers directly, use it.
                # Otherwise, you need a mapping like:
                label_mapping = {
                    'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3, 'motorcycle': 4,
                    'bicycle': 5, 'pedestrian': 6, 'traffic_light': 7, 'construction_vehicle': 8,
                    'barrier': 9, 'animal': 10, 'other_vehicle': 11, 'sign': 12
                }
                # Assuming 'detection_name' exists and maps to an integer label
                label = label_mapping.get(det['detection_name'], -1) # Default to -1 if not found
                labels_list.append(label)

            # Convert lists to PyTorch tensors
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64) # Use int64 for labels

            # Create LiDARInstance3DBoxes object
            # Note: LiDARInstance3DBoxes constructor expects a tensor,
            # which will be the 'boxes_3d' part of your desired PKL structure.
            lidar_boxes = LiDARInstance3DBoxes(boxes_tensor)
            print(lidar_boxes)

            # Append the structured data for this sample
            processed_data.append({
                'boxes_3d': lidar_boxes,
                'scores_3d': scores_tensor,
                'labels_3d': labels_tensor
            })

        # Save the processed data to a pickle file
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Successfully created and saved structured data to '{pkl_filepath}'.")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_filepath}': {e}")
        print("Please ensure the JSON file is correctly formatted.")
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please ensure 'torch', 'mmdet3d', and 'scipy' are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    input_json_file = "../submission/pts_bbox/results_nusc.json"
    output_pkl_file = "../submission/pts_bbox/results_nusc_streamPETR_generated.pkl"

    # --- Perform the conversion ---
    convert_json_to_structured_pkl(input_json_file, output_pkl_file)

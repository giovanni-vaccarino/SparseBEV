import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R # Required for quaternion conversions

def merge_ground_truth_into_results(results_nusc_path, gt_data_path, output_path):
    """
    Merges ground truth data into the results_nusc.json structure.

    The ground truth data from gt_data_path is converted to match the
    format of predictions (translation, size, rotation, detection_name)
    and added under a new 'ground_truth_results' key in the results_nusc.json.

    Args:
        results_nusc_path (str): Path to the results_nusc.json file (predicted labels).
        gt_data_path (str): Path to the ground truth JSON file (e.g., 2025-03-13_11-54_000000.json).
        output_path (str): Path where the merged JSON file will be saved.
    """
    if not os.path.exists(results_nusc_path):
        print(f"Error: results_nusc.json not found at '{results_nusc_path}'.")
        return
    if not os.path.exists(gt_data_path):
        print(f"Error: Ground truth JSON not found at '{gt_data_path}'.")
        return

    try:
        # Load the results_nusc.json (predicted labels)
        with open(results_nusc_path, 'r') as f:
            results_nusc_data = json.load(f)
        print(f"Successfully loaded '{results_nusc_path}'.")

        # Load the ground truth JSON (e.g., 2025-03-13_11-54_000000.json)
        with open(gt_data_path, 'r') as f:
            gt_data = json.load(f)
        print(f"Successfully loaded '{gt_data_path}'.")

        # Prepare the structure for ground truth results
        # Assuming the ground truth file is for a single sample token,
        # which can be inferred from its filename if consistent, or explicitly passed.
        # For this example, let's assume the sample token is "2025-03-13_11-54_000000"
        # as seen in your previous context.
        sample_token_for_gt = os.path.splitext(os.path.basename(gt_data_path))[0]
        # In case the sample token in the GT file itself is structured differently,
        # you might need to manually set or derive it. For now, using filename.

        ground_truth_results = {sample_token_for_gt: []}

        # Define a mapping from ground truth 'obj_type' to a 'detection_name'
        # This mapping is crucial and should align with your dataset's conventions.
        # Add or modify this mapping based on your specific object types.
        label_type_to_detection_name = {
            "Car": "car",
            "Truck": "truck",
            "Bus": "bus",
            "Pedestrian": "pedestrian",
            "Bicycle": "bicycle",
            "Motorcycle": "motorcycle",
            "Trailer": "trailer",
            "ConstructionVehicle": "construction_vehicle",
            "Barrier": "barrier",
            "TrafficCone": "traffic_cone", # Often mapped to traffic_light or other
            "BicycleRider": "pedestrian", # Example: mapping a rider to pedestrian
            # Add more mappings as per your dataset's `obj_type` and desired `detection_name`
        }


        # Process each ground truth object
        for obj in gt_data:
            obj_type = obj['obj_type']
            psr = obj['psr']

            # Extract position (translation)
            translation = [psr['position']['x'], psr['position']['y'], psr['position']['z']]

            # Extract scale (size)
            # Assuming size in results_nusc.json is [length, width, height]
            size = [psr['scale']['x'], psr['scale']['y'], psr['scale']['z']]

            # Extract rotation (yaw from z-component, convert to quaternion [x, y, z, w])
            # Assuming psr.rotation.z is yaw in radians (rotation around Z-axis)
            yaw = psr['rotation']['z']
            # For a rotation around the Z-axis only, quaternion is [0, 0, sin(yaw/2), cos(yaw/2)]
            # The order in results_nusc.json's rotation is [x, y, z, w]
            rotation_quat_x = 0.0
            rotation_quat_y = 0.0
            rotation_quat_z = np.sin(yaw / 2.0)
            rotation_quat_w = np.cos(yaw / 2.0)
            rotation = [rotation_quat_x, rotation_quat_y, rotation_quat_z, rotation_quat_w]

            # Get detection name
            detection_name = label_type_to_detection_name.get(obj_type, "unknown")
            if detection_name == "unknown":
                print(f"Warning: Unknown obj_type '{obj_type}' encountered. Mapped to 'unknown'.")

            # Create a dictionary for the ground truth object, mimicking prediction format
            gt_obj_formatted = {
                "sample_token": sample_token_for_gt,
                "translation": translation,
                "size": size,
                "rotation": rotation,
                "velocity": [0.0, 0.0], # Ground truth might not have velocity, set to zero or omit
                "detection_name": detection_name,
                "detection_score": 1.0, # Ground truth typically has a score of 1.0
                "attribute_name": "ground_truth" # Add a unique attribute to identify as GT
            }
            ground_truth_results[sample_token_for_gt].append(gt_obj_formatted)

        # Add the ground_truth_results to the main results_nusc_data
        results_nusc_data['ground_truth_results'] = ground_truth_results

        # Save the merged data to a new JSON file
        with open(output_path, 'w') as f:
            json.dump(results_nusc_data, f, indent=2) # Use indent for pretty printing
        print(f"Merged data successfully saved to '{output_path}'.")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}. Please check your input JSON files for syntax errors.")
    except KeyError as e:
        print(f"Key error: Missing expected key '{e}'. Please check the structure of your JSON files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your existing results_nusc.json
    results_nusc_file = "../submission/pts_bbox/empty.json"
    # Path to your ground truth JSON for a single frame
    ground_truth_file = "../label/2025-03-13_11-54_000040.json"
    # Desired output path for the merged JSON file
    output_merged_json_file = "results_nusc_with_gt.json"

    # --- Perform the merge ---
    merge_ground_truth_into_results(results_nusc_file, ground_truth_file, output_merged_json_file)
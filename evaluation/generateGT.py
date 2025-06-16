import os
import json
import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw in radians) to a quaternion [w, x, y, z].
    """
    # Calculate cosine and sine for each half-angle
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def create_gt_from_labels(label_folder, output_file, number_of_samples):
    """
    Generates a NuScenes-like ground truth file from a folder of label files.

    Args:
        label_folder (str): The path to the folder containing the label JSON files.
        output_file (str): The path where the final JSON file will be saved.
    """
    # Mapping from the input obj_type to the NuScenes detection_name
    TYPE_MAPPING = {
        "Car": "car",
        "Pedestrian": "pedestrian",
        "BicycleRider": "bicycle",
        "Bicycle": "bicycle",  # Added mapping for "Bicycle"
        "Truck": "truck",
        "Bus": "bus",
        # Add any other mappings you might have from your source data
    }

    ATTRIBUTE_MAPPING = {
        "Car": "vehicle.parked",
        "Pedestrian": "pedestrian.standing",
        "BicycleRider": "cycle.without_rider",
        "Bicycle": "cycle.without_rider",  # Added mapping for "Bicycle"
    }   

    final_output = {
        "meta": {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "results": {},
    }

    # Iterate from sample 0 to 320
    for i in range(number_of_samples):
        sample_index_str = f"{i:06d}"
        sample_token = f"2025-03-13_11-54_{sample_index_str}"
        file_path = os.path.join(label_folder, f"{sample_token}.json")

        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        try:
            with open(file_path, 'r') as f:
                source_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
            continue

        detections_for_sample = []
        for obj in source_data:
            obj_type = obj.get("obj_type")
            psr = obj.get("psr")

            if not obj_type or not psr:
                continue

            # Map the object type to a valid NuScenes category name
            detection_name = TYPE_MAPPING.get(obj_type)
            attribute_name = ATTRIBUTE_MAPPING.get(obj_type)

            if not detection_name:
                print(f"Warning: Unknown obj_type '{obj_type}' in {file_path}. Skipping object.")
                continue
            
            rotation_data = psr.get("rotation", {})
            roll = rotation_data.get("x", 0.0)
            pitch = rotation_data.get("y", 0.0)
            yaw = rotation_data.get("z", 0.0)


            # Transform the data into the NuScenes format
            transformed_obj = {
                "sample_token": sample_token,
                "translation": [
                    psr["position"]["x"],
                    psr["position"]["y"],
                    psr["position"]["z"],
                ],
                # Assuming scale.y is width, scale.x is length, scale.z is height
                "size": [
                    psr["scale"]["y"],
                    psr["scale"]["x"],
                    psr["scale"]["z"],
                ],
                "rotation": euler_to_quaternion(roll, pitch, yaw),
                "velocity": [0.0, 0.0],  # No velocity data in source
                "detection_name": detection_name,
                "detection_score": 1.0,  # Ground truth has a score of 1.0
                "attribute_name": attribute_name,
            }
            detections_for_sample.append(transformed_obj)
        
        # Add the list of detections for this sample to the main results dictionary
        final_output["results"][sample_token] = detections_for_sample
        print(f"Processed: {sample_token}.json")

    # Write the aggregated results to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        print(f"\nSuccessfully generated ground truth file at: {output_file}")
    except IOError as e:
        print(f"\nError writing to file: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # The folder where your source JSON files are located
    LABELS_INPUT_DIR = "label_sustech"
    # The name of the final output file
    GT_OUTPUT_FILE = "results_nusc_with_gt.json"
    number_of_samples = 80  # Adjust this if you have more or fewer samples
    
    # --- Execution ---
    if not os.path.isdir(LABELS_INPUT_DIR):
        print(f"Error: Input directory '{LABELS_INPUT_DIR}' not found.")
        print("Please create the directory and place your label files inside.")
    else:
        create_gt_from_labels(LABELS_INPUT_DIR, GT_OUTPUT_FILE, number_of_samples)


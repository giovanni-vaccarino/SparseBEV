import json
import math
from pathlib import Path
from typing import Dict, List

def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Convert a quaternion to yaw rotation angle around the z-axis.
    """
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def convert_object(obj: Dict) -> Dict:
    """
    Convert a single object to mmdet3d format.
    """
    # Reorder size from [w, l, h] to [l, w, h]
    size = [obj['size'][1], obj['size'][0], obj['size'][2]]
    
    # Convert quaternion to yaw angle
    rotation = quaternion_to_yaw(
        obj['rotation'][0],
        obj['rotation'][1],
        obj['rotation'][2],
        obj['rotation'][3]
    )
    
    return {
        'sample_token': obj['sample_token'],
        'translation': obj['translation'],
        'size': size,
        'rotation': rotation,
        'velocity': obj['velocity'],
        'detection_name': obj['detection_name'],
        'detection_score': obj['detection_score'],
        'attribute_name': obj['attribute_name']
    }

def process_input_json(input_path: str) -> List[Dict]:
    """
    Process the input JSON file and convert all objects to mmdet3d format.
    """
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    converted_objects = []
    
    # Check if the input has the expected structure
    if 'results' in input_data:
        for sample_token, objects in input_data['results'].items():
            for obj in objects:
                converted_objects.append(convert_object(obj))
    else:
        # If the input is already a list of objects
        for obj in input_data:
            converted_objects.append(convert_object(obj))
    
    return converted_objects

def save_output_json(output_path: str, data: List[Dict]):
    """
    Save the converted data to a JSON file.
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    # Path to your input JSON file
    input_json_path = '../submission/pts_bbox/results_nusc.json'  # Change this to your input file path
    
    # Output path
    output_json_path = '../submission/pts_bbox/results_nusc_mmdet3d.json'  # Change this to your desired output file path
    
    # Process the input and save the output
    converted_data = process_input_json(input_json_path)
    save_output_json(output_json_path, converted_data)
    
    print(f"Successfully converted data. Output saved to {output_json_path}")

if __name__ == '__main__':
    main()
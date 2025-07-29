import pickle
import json
from scipy.spatial.transform import Rotation as R 
import numpy as np

result_to_fill = []
# Convert the json to results_nuscenes format
def convert_to_nuscenes_format():

    for i in range(0, 20):
        ground_truth = f"../label/2025-03-13_11-54_{i:06d}.json"
        with open(ground_truth, "r") as f:
            gt = json.load(f)
        result = gt[i]
        psr = result["psr"]

        rotation = [psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]]

        #print(rotation)
        r = R.from_euler('xyz', rotation, degrees=True)
        rotation_matrix = r.as_quat()
        print(rotation_matrix)
        # Invert the rotation matrix to get the quaternion
        #inverted_rotation_matrix = np.linalg.inv(rotation_matrix)

        rotation_final = [rotation_matrix[3], rotation_matrix[0], rotation_matrix[1], rotation_matrix[2]]
        print(rotation_final)

        new_object = {
            "sample_token": f"2025-03-13_11-54_{i:06d}",
            "translation": [psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]],
            "size": [psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]],
            "rotation": rotation_final,
            "velocity": [0.0,0.0],
            "detection_name" : "ground_truth",
            "detection_score": 0.99,
            "attribute_name": "ground_truth",
        }

        result_to_fill.append(new_object)

    return result_to_fill
        

results = convert_to_nuscenes_format()
        
#print(results)

with open ('../submission/pts_bbox/results_nusc.json', 'r') as f:
    results_to_add = json.load(f)
    for res in results:
        results_to_add["results"][res["sample_token"]].append(res)

with open ('../submission/pts_bbox/results_nusc_gt.json', 'w') as f:
    json.dump(results_to_add, f, indent=4)
    print("Added ground truth to results_nusc_gt.json")

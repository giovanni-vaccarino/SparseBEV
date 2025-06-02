import torch
import json
import numpy as np
from PIL import Image
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from scipy.spatial.transform import Rotation as R
from visualize_3d_carlo import draw_bboxes_3d_image
import pickle


def nusc_to_mmdet3d_boxes(nusc_detections):
    boxes = []
    labels = []
    for obj in nusc_detections:
        x, y, z = obj['translation']
        w, l, h = obj['size']  # MMDet expects [l, w, h]
        quat = obj['rotation']  # [w, x, y, z]
        label_name = obj['detection_name']
        
        # Convert quaternion to yaw (heading angle)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
        yaw = r.as_euler('zyx')[0]  # Extract Z-axis rotation

        boxes.append([x, y, z, l, w, h, yaw])

        # Map label to int — make sure this matches color_dict in visualizer
        label_id = {
            "car": 0,
            "pedestrian": 1,
            # add more if needed
        }.get(label_name, 2)  # unknown = 2

        labels.append(label_id)
        
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    return LiDARInstance3DBoxes(boxes_tensor), labels_tensor

output_json_path = '../submission/pts_bbox/results_nusc.json'

for camera in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
    for number_image in range(0,100):
        # === Paths ===
        image_path = f'../data/aida_mdp_500/samples/{camera}/2025-03-13_11-54_{number_image:06d}.png'

        with open(output_json_path, 'r') as f:
            results = json.load(f)

        sample_token = f"2025-03-13_11-54_{number_image:06d}"
        detections = results['results'][sample_token]
        # === Convert to MMDet3D boxes ===
        #bboxes_3d, labels = nusc_to_mmdet3d_boxes(detections)
        
        # === Load image ===
        image = np.array(Image.open(image_path).convert("RGB"))

        # === getting info from the .pkl  ===
        with open('../data/aida_mdp_500/test_mdp_aida500.pkl', 'rb') as f:
            data = pickle.load(f)

        # === Pick the sample by token ===
        info = next(i for i in data['infos'] if i['token'] == sample_token)
        cam_info = info['cams'][camera]  # choose camera

        # === Get intrinsics (P) ===
        cam_intrinsic = np.array(cam_info['cam_intrinsic'])  # 3x3
        P = np.eye(3, 4)
        P[:3, :3] = cam_intrinsic
        P = torch.tensor(P, dtype=torch.float32)

        # === Get extrinsics: LiDAR → Camera ===
        # The stored values are camera → LiDAR, so invert them

        R_cam2lidar = np.array(cam_info['sensor2lidar_rotation'])  # 3x3
        t_cam2lidar = np.array(cam_info['sensor2lidar_translation'])  # 3x1

        # Invert the transform to get LiDAR → Camera
        R_lidar2cam = R_cam2lidar.T
        t_lidar2cam = -R_lidar2cam @ t_cam2lidar

        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[:3, :3] = R_lidar2cam
        Tr_velo_to_cam[:3, 3] = t_lidar2cam
        Tr_velo_to_cam = torch.tensor(Tr_velo_to_cam, dtype=torch.float32)

        # === Rectification matrix (identity) ===
        R_rect = torch.eye(4)

        # === MMDet bboxes ===

        with open("../submission/pts_bbox/results.pkl", "rb") as f:
            results = pickle.load(f)

        results_filtered = []

        filter_bboxes = results[number_image]['scores_3d'] > 0.3

        bboxes_3d_manual = results[number_image]['boxes_3d'][filter_bboxes]
        labels_3d_manual = results[number_image]['labels_3d'][filter_bboxes].numpy()

        color_dict = {
            0: (255, 255, 0),   # Yellow
            1: (0, 255, 0),     # Green
            2: (255, 0, 0),     # Red
            3: (150, 150, 0), 
            4: (100, 100, 100), 
            5: (0, 160, 100), 
            6: (0, 255, 255),   # Cyan
            7: (255, 0, 255),   # Magenta
            8: (255, 165, 0),   # Orange
            9: (128, 0, 255),    # Violet/Purple
            10: (100, 50, 100)
        }


        # === Visualize ===
        output_image = draw_bboxes_3d_image(
            image=image,
            bboxes_3d=bboxes_3d_manual,
            labels=labels_3d_manual,
            projection_matrix=P,
            rectification_matrix=None,
            tr_velo_to_cam=Tr_velo_to_cam,
            color_dict=color_dict,
            lidar_coords=True
        )

        output_image.save(f"output_images/{camera}/output_with_boxes_{camera}_{number_image}.png")



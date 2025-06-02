import torch
import json
import numpy as np
from PIL import Image
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from scipy.spatial.transform import Rotation as R
from visualize_3d_carlo import show_bboxes_3d
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np


# === Paths ===
output_json_path = '../submission/pts_bbox/results_nusc.json'
pcd_path = '../point_cloud/2025-03-13_11-54_000000.pcd'

# === getting info from the .pkl  ===
with open('../data/aida_mdp_500/test_mdp_aida500.pkl', 'rb') as f:
    data = pickle.load(f)


# === Pick the sample by token ===
sample_token = "2025-03-13_11-54_000000"
info = next(i for i in data['infos'] if i['token'] == sample_token)
cam_info = info['cams']['CAM_FRONT']  # choose camera

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

#pcd = []

#with open(pcd_path, "r") as f:
#    for line in f:
#        x = line.split(" ")[0]
#        y = line.split(" ")[1]
#        z = line.split(" ")[2]
#        pcd.append([float(x), float(y), float(z)])

#pcd_0 = np.array(pcd)
#print(pcd_0)

#print(pcd_0.shape)

# Retrieve the point cloud from the .pcd file


# Read the .pcd (ASCII or binary) into an Open3D PointCloud
pcd_o3d = o3d.io.read_point_cloud(pcd_path)

# Extract Nx3 pts, then append a 1 for homogeneous coords ⇒ Nx4
pts = np.asarray(pcd_o3d.points)                   # shape (N,3)
scan = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1).astype(np.float32)  # shape (N,4)


results_filtered = []

filter_bboxes = results[0]['scores_3d'] > 0.25
print(filter_bboxes.shape)
#bboxes_3d_manual = results[0]['boxes_3d'][filter_bboxes]
# Slice and re-wrap the bounding boxes correctly
boxes_tensor = results[0]['boxes_3d'][filter_bboxes].tensor[:, :7]
bboxes_3d_manual = LiDARInstance3DBoxes(boxes_tensor)

print(bboxes_3d_manual)

labels_3d_manual = results[0]['labels_3d'][filter_bboxes].numpy()

color_dict = {
    0: (255, 255, 0),   # Yellow
    1: (0, 255, 0),     # Green
    2: (255, 0, 0),     # Red
    6: (0, 255, 255),   # Cyan
    7: (255, 0, 255),   # Magenta
    8: (255, 165, 0),   # Orange
    9: (128, 0, 255)    # Violet/Purple
}

calibration = {
    'Tr_velo_to_cam': Tr_velo_to_cam.numpy()
}

# === Visualize ===
output_image = show_bboxes_3d(
    point_cloud=scan, calibration_data=calibration,
    bboxes_3d=bboxes_3d_manual,
    labels=labels_3d_manual,
    colors_dict=color_dict,
    save_capture=True,
    adjust_position=False
)

plt.imshow(output_image)
plt.axis('off')
plt.show()

output_image.save("output_with_boxes.png")

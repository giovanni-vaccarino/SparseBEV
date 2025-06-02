import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import math

def scale_intrinsic(K, original_size, target_size):
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled






# gt_boxes -> array con elementi da 7 delle coord

#gt_names -> cosa sono(pedestrian, car, …) fare il mapping di tutt i nomi

# gt_velocity -> array di array di 2 elementi (0)

# num_lidar_pts -> array per ogni box di quanto ha

# num_radad_ar_pts -> array per ogni box di quanto ha

# add field valid_flag -> with all true 
    
# filepath: /home/ec_500_a2a/AIDA/ScriptToCreatePKL/genTrainPkl.py

def load_ground_truth(frame_token):
    """
    Loads ground truth boxes and labels from a JSON file.
    Returns:
        gt_boxes: list of [x, y, z, dx, dy, dz, yaw]
        gt_names: list of int (class ids)
    """
    # Map your object types to integer class ids
    class_map = {
        "Car": "car",
        "Pedestrian": "pedestrian",
        "BicycleRider": "bicycle",
        # Add more classes as needed
    }

    json_path = f"../SparseBEV/label_sustech/{frame_token}.json"
    gt_boxes = []
    gt_names = []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not load {json_path}: {e}")
        return [], []

    for obj in data:
        try:
            if "psr" not in obj or "obj_type" not in obj:
                continue
            psr = obj["psr"]
            obj_type = obj["obj_type"]
            if obj_type not in class_map:
                continue

            x = psr["position"]["x"]
            y = psr["position"]["y"]
            z = psr["position"]["z"]
            dx = psr["scale"]["x"]
            dy = psr["scale"]["y"]
            dz = psr["scale"]["z"]
            # Assuming rotation is in degrees and yaw is around Z axis
            yaw_deg = psr["rotation"]["z"]
            yaw = math.radians(yaw_deg)

            gt_boxes.append([x, y, z, dx, dy, dz, yaw])
            gt_names.append(class_map[obj_type])
        except Exception as e:
            print(f"Skipping malformed object in {json_path}: {e}")
            continue

    return gt_boxes, gt_names

def generate_my_infos(dataset_root, output_file, frames):
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    cam_intrinsic_before_scaling = np.array([
        [1014.166, 0.0, 717.96],
        [0.0, 1001.4649999999999, 506.69000000000005],
        [0.0, 0.0, 1.0]
    ])

    original_size = (1440, 930)
    target_size = (1600, 900)

    cam_intrinsic = scale_intrinsic(cam_intrinsic_before_scaling, original_size, target_size)

    CAM_FRONT= [
        [-0.015262883957516097, -0.9998640836938528, -0.006233659627504093, 0.006523329060158488],
        [-0.029473612607531223,  0.006681572724141494, -0.9995432270521337, -0.1247231921362766],
        [ 0.999449023479017, -0.015072183816029617, -0.02957158673211907, -0.03971285908667528],
        [ 0, 0, 0, 1]
    ]

    CAM_FRONT_LEFT= [
        [ 0.9395867699697493, -0.34169094845212017,  0.02059119820952398, 0.02787550120945102],
        [ 0.010782806389290252, -0.03058023102560659, -0.999474152020347, -0.1267788775357696],
        [ 0.3421409545555522,  0.9393147210686689, -0.025048393152146232, -0.14532624460741952],
        [ 0, 0, 0, 1]
    ]

    CAM_FRONT_RIGHT= [
        [-0.941188686807867, -0.3378813948131003, -0.00013734640977485356, -0.020941806952435867],
        [-0.01518054305797053,  0.04269237949289996, -0.998972928484901, -0.11948460921111645],
        [0.3375402301020541, -0.940219933724226, -0.04531080765407303, -0.1451091712091093],
        [ 0, 0, 0, 1]
    ]

    INV_CAM_FRONT = np.linalg.inv(CAM_FRONT)
    INV_CAM_FRONT_LEFT = np.linalg.inv(CAM_FRONT_LEFT)
    INV_CAM_FRONT_RIGHT = np.linalg.inv(CAM_FRONT_RIGHT)
    
    extrinsics = {
        'CAM_FRONT':INV_CAM_FRONT[:3, :3].tolist(),
        'CAM_FRONT_LEFT': INV_CAM_FRONT_LEFT[:3, :3].tolist(),
        'CAM_FRONT_RIGHT': INV_CAM_FRONT_RIGHT[:3, :3].tolist(),
        'CAM_BACK': INV_CAM_FRONT[:3, :3].tolist(),
        'CAM_BACK_LEFT': INV_CAM_FRONT[:3, :3].tolist(),
        'CAM_BACK_RIGHT': INV_CAM_FRONT[:3, :3].tolist(),
    }

    translations = {
        'CAM_FRONT': INV_CAM_FRONT[:3, 3].tolist(),
        'CAM_FRONT_LEFT': INV_CAM_FRONT_LEFT[:3, 3].tolist(),
        'CAM_FRONT_RIGHT': INV_CAM_FRONT_RIGHT[:3, 3].tolist(),
        'CAM_BACK': [0.00652333, -0.12511408, -0.03846366],
        'CAM_BACK_LEFT': [0.0278755, -0.12677888, -0.14532624],
        'CAM_BACK_RIGHT': [-0.02094181, -0.11948461, -0.14510917]
    }

    infos = []

    # Setted to default values since we do not count global
    identity_quat = [1.0, 0.0, 0.0, 0.0]
    identity_rot_matrix = R.from_quat([identity_quat[1], identity_quat[2], identity_quat[3], identity_quat[0]]).as_matrix().tolist()
    zero_translation = [0.0, 0.0, 0.0]

    for i, frame in enumerate(frames):
        # Using this as a token 
        frame_token = os.path.splitext(frame)[0]
        cams = {}
        for cam in cam_names:
            cam_path = os.path.join('data/aida_mdp_500/samples', cam, frame)
            cams[cam] = {
                'data_path': cam_path,
                'sensor2ego_translation': translations[cam],
                'sensor2ego_rotation': extrinsics[cam],
                'sensor2global_translation': zero_translation,
                'sensor2global_rotation': identity_rot_matrix, 
                'sensor2lidar_translation': translations[cam], 
                'sensor2lidar_rotation': extrinsics[cam],
                'cam_intrinsic': cam_intrinsic,
                'timestamp': 1000000 
            }

        num_sweeps = 7
        sweeps = []
        for j in range(1, num_sweeps + 1):
            if i - j < 0:
                continue
            sweep_frame = frames[i - j]
            sweep_cams = {}
            for cam in cam_names:
                sweep_cams[cam] = {
                    'data_path': os.path.join('data/aida_mdp_500/samples', cam, sweep_frame),
                    'sensor2ego_translation': translations[cam],
                    'sensor2ego_rotation': extrinsics[cam],
                    'sensor2global_translation': zero_translation,
                    'sensor2global_rotation': identity_rot_matrix,  
                    'cam_intrinsic': cam_intrinsic,
                    'timestamp': 1000000 - j * 100000
                }
            sweep_flat = sweep_cams.copy()
            sweep_flat['timestamp'] = 1000000 - j * 100000
            sweeps.append(sweep_flat)
            
        gt_boxes, gt_labels = load_ground_truth(frame_token)
        

        info = {
            'token': frame_token,
            'timestamp': 1000000,
            'cams': cams,
            'sweeps': sweeps,
            'ego2global_translation': zero_translation,
            'ego2global_rotation': identity_quat,
            'lidar2ego_translation': zero_translation,
            'lidar2ego_rotation': identity_quat,
            'gt_boxes': np.array(gt_boxes, dtype=np.float32),
            'gt_names': np.array(gt_labels),
            'gt_velocity': np.array(np.zeros((len(gt_boxes), 2), dtype=np.float32)),
            'num_lidar_pts': np.array(np.zeros((len(gt_boxes),), dtype=np.int64)),
            'num_radar_pts': np.array(np.zeros((len(gt_boxes),), dtype=np.int64)),
            'valid_flag': np.array(np.ones((len(gt_boxes),), dtype=np.bool_)),
        }

        infos.append(info)

    output = {
        'infos': infos,
        'metadata': {
            'version': 'aida_v1.0' 
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

def __main__():
    dataset_root = 'data/aida_mdp_500'
    all_frames = sorted(os.listdir(os.path.join(dataset_root, 'CAM_FRONT')))
    num_train = int(0.8 * len(all_frames))
    train_frames = all_frames[:num_train]
    val_frames = all_frames[num_train:]

    generate_my_infos(dataset_root, 'train_mdp_aida500.pkl', train_frames)
    generate_my_infos(dataset_root, 'val_mdp_aida500.pkl', val_frames)

if __name__ == "__main__":
    __main__()
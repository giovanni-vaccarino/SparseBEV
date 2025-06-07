import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from .config import ORIGINAL_SIZE, TARGET_SIZE, CAM_INTRINSIC_UNSCALED, CAM_EXTRINSICS, CAM_BACK_TRANSLATIONS, DEFAULT_INV_ROTATION

def scale_intrinsic(K, original_size, target_size):
    """
    Scales the camera intrinsic matrix K from the original image size to the target size.

    Args:
        K (np.ndarray): Original 3x3 intrinsic matrix.
        original_size (tuple): (width, height) of the original image.
        target_size (tuple): (width, height) of the target resized image.

    Returns:
        np.ndarray: Scaled 3x3 intrinsic matrix.
    """
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[1, 2] *= scale_y

    return K_scaled

def generate_my_infos(dataset_root, output_file):
    """
    Generates per-frame metadata info for camera-based 3D perception datasets.

    Args:
        dataset_root (str): Root directory containing image sequences (organized by camera name).
        output_file (str): Path to the output .pkl file where the metadata will be saved.
    """
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    cam_intrinsic = scale_intrinsic(CAM_INTRINSIC_UNSCALED, ORIGINAL_SIZE, TARGET_SIZE)

    extrinsics = {}
    translations = {}

    for cam in cam_names:
        if cam in CAM_EXTRINSICS:
            inv_pose = np.linalg.inv(CAM_EXTRINSICS[cam])
            extrinsics[cam] = inv_pose[:3, :3].tolist()
            translations[cam] = inv_pose[:3, 3].tolist()
        else:
            extrinsics[cam] = DEFAULT_INV_ROTATION.tolist()
            translations[cam] = CAM_BACK_TRANSLATIONS[cam]

    frames = sorted(os.listdir(os.path.join(dataset_root, 'CAM_FRONT')))
    identity_quat = [1.0, 0.0, 0.0, 0.0]
    identity_rot_matrix = R.from_quat([0.0, 0.0, 0.0, 1.0]).as_matrix().tolist()
    zero_translation = [0.0, 0.0, 0.0]

    infos = []

    for i, frame in enumerate(frames):
        frame_token = os.path.splitext(frame)[0]
        cams = {}

        for cam in cam_names:
            cams[cam] = {
                'data_path': os.path.join('data/aida_mdp_500/samples', cam, frame),
                'sensor2ego_translation': translations[cam],
                'sensor2ego_rotation': extrinsics[cam],
                'sensor2global_translation': zero_translation,
                'sensor2global_rotation': identity_rot_matrix,
                'sensor2lidar_translation': translations[cam],
                'sensor2lidar_rotation': extrinsics[cam],
                'cam_intrinsic': cam_intrinsic,
                'timestamp': 1000000
            }

        sweeps = []
        for j in range(1, 8):
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

        infos.append({
            'token': frame_token,
            'timestamp': 1000000,
            'cams': cams,
            'sweeps': sweeps,
            'ego2global_translation': zero_translation,
            'ego2global_rotation': identity_quat,
            'lidar2ego_translation': zero_translation,
            'lidar2ego_rotation': identity_quat
        })

    output = {
        'infos': infos,
        'metadata': {
            'version': 'aida_v1.0'
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    generate_my_infos('data/aida_mdp_500', 'test_mdp_aida500.pkl')

import numpy as np

# Original and target image sizes
ORIGINAL_SIZE = (1440, 930)
TARGET_SIZE = (1600, 900)

# Unscaled intrinsic matrix
CAM_INTRINSIC_UNSCALED = np.array([
    [1014.166, 0.0, 717.96],
    [0.0, 1001.465, 506.69],
    [0.0, 0.0, 1.0]
])

# Camera extrinsics (camera to ego) 4x4 matrices
CAM_EXTRINSICS = {
    'CAM_FRONT': np.array([
        [-0.01526288, -0.99986408, -0.00623366, 0.00652333],
        [-0.02947361,  0.00668157, -0.99954323, -0.12472319],
        [ 0.99944902, -0.01507218, -0.02957159, -0.03971286],
        [ 0, 0, 0, 1]
    ]),
    'CAM_FRONT_LEFT': np.array([
        [ 0.93958677, -0.34169095,  0.0205912,  0.0278755],
        [ 0.01078281, -0.03058023, -0.99947415, -0.12677888],
        [ 0.34214095,  0.93931472, -0.02504839, -0.14532624],
        [ 0, 0, 0, 1]
    ]),
    'CAM_FRONT_RIGHT': np.array([
        [-0.94118869, -0.33788139, -0.00013735, -0.02094181],
        [-0.01518054,  0.04269238, -0.99897293, -0.11948461],
        [ 0.33754023, -0.94021993, -0.04531081, -0.14510917],
        [ 0, 0, 0, 1]
    ])
}

# Default extrinsics for missing cameras (reused)
DEFAULT_INV_ROTATION = np.linalg.inv(CAM_EXTRINSICS['CAM_FRONT'])[:3, :3]
DEFAULT_INV_TRANSLATION = [0.00652333, -0.12511408, -0.03846366]
CAM_BACK_TRANSLATIONS = {
    'CAM_BACK': DEFAULT_INV_TRANSLATION,
    'CAM_BACK_LEFT': [0.0278755, -0.12677888, -0.14532624],
    'CAM_BACK_RIGHT': [-0.02094181, -0.11948461, -0.14510917],
}

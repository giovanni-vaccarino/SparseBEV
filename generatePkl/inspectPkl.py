import pickle

pkl_path = '../SparseBEV/data/aida_mdp_500/test_mdp_aida500.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

bad_sweeps = []

for i, info in enumerate(data['infos']):
    for j, sweep in enumerate(info['sweeps']):
        if 'cams' not in sweep:
            bad_sweeps.append((i, j))
            print(f"[❌] Frame {i}, Sweep {j}: NO 'cams' key, top-level keys = {list(sweep.keys())}")
        else:
            missing_cams = []
            for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                if cam not in sweep['cams']:
                    missing_cams.append(cam)
            if missing_cams:
                print(f"[⚠️] Frame {i}, Sweep {j}: Missing cameras in 'cams': {missing_cams}")

print("\n==== Summary ====")
if not bad_sweeps:
    print("✅ All sweeps correctly contain 'cams'.")
else:
    print(f"❌ Found {len(bad_sweeps)} sweeps without 'cams'. Indices: {bad_sweeps}")


# Read the one line of the pkl file
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
    print(data)
    # Print the first 5 elements of the 'infos' list
    for i in range(5):
        print(data['infos'][i])
import json
import numpy as np
from scipy.spatial.transform import Rotation
from collections import defaultdict

# Class-wise matching distance thresholds (nuScenes-style)
CLASS_THRESHOLDS = {
    'car': 2.0,
    'truck': 2.0,
    'bus': 2.0,
    'trailer': 2.0,
    'construction_vehicle': 2.0,
    'pedestrian': 0.5,
    'motorcycle': 1.0,
    'bicycle': 1.0,
    'traffic_cone': 0.5,
    'barrier': 0.5
}

def load_results(path):
    """Loads detection results from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)["results"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading '{path}': {e}")
        return None

def group_by_sample_and_class(results):
    """Groups detections by sample_token and then by detection_name."""
    grouped = defaultdict(lambda: defaultdict(list))
    if results is None:
        return grouped
    for sample_token, detections in results.items():
        for det in detections:
            grouped[sample_token][det['detection_name']].append(det)
    return grouped

def quaternion_yaw_angle_diff(q1, q2):
    """Calculates the smallest angle difference in yaw between two quaternions."""
    yaw1 = Rotation.from_quat(q1).as_euler('zyx', degrees=False)[0]
    yaw2 = Rotation.from_quat(q2).as_euler('zyx', degrees=False)[0]
    diff = ((yaw1 - yaw2 + np.pi) % (2 * np.pi)) - np.pi
    return abs(diff)

def aligned_iou(size1, size2):
    """Calculates the 3D IoU for two axis-aligned bounding boxes."""
    dims1, dims2 = np.array(size1), np.array(size2)
    inter = np.prod(np.minimum(dims1, dims2))
    union = np.prod(dims1) + np.prod(dims2) - inter
    return inter / union if union > 0 else 0.0

def calculate_ap(matches, total_gts):
    """
    Calculates Average Precision (AP) using 11-point interpolation.
    Args:
        matches (list): A list of tuples (score, is_true_positive).
        total_gts (int): The total number of ground truth objects for this class.
    Returns:
        float: The Average Precision for this class.
    """
    if total_gts == 0:
        return float('nan')
    if not matches:
        return 0.0

    # Sort matches by detection score in descending order
    matches.sort(key=lambda x: x[0], reverse=True)

    tp_count = 0
    fp_count = 0
    precision_vals = []
    recall_vals = []

    for _, is_tp in matches:
        if is_tp:
            tp_count += 1
        else:
            fp_count += 1
        
        precision = tp_count / (tp_count + fp_count)
        recall = tp_count / total_gts
        precision_vals.append(precision)
        recall_vals.append(recall)

    # Use 11-point interpolation for AP calculation
    ap = 0.0
    for r_level in np.linspace(0, 1, 11):
        try:
            # Find the maximum precision for recall values >= r_level
            p = max(p for p, r in zip(precision_vals, recall_vals) if r >= r_level)
        except ValueError:
            p = 0.0
        ap += p / 11.0
        
    return ap

def evaluate(pred_results, gt_results):
    """
    Performs a comprehensive evaluation including mAP and other nuScenes metrics.
    """
    pred_by_sample = group_by_sample_and_class(pred_results)
    gt_by_sample = group_by_sample_and_class(gt_results)

    # --- Data Aggregation ---
    errors = defaultdict(lambda: defaultdict(list))
    class_matches = defaultdict(list) # To store (score, is_tp) for AP calculation
    total_gts_per_class = defaultdict(int)
    
    # Pre-calculate total number of ground truths per class
    for sample_data in gt_by_sample.values():
        for cls, gts in sample_data.items():
            if cls in CLASS_THRESHOLDS:
                total_gts_per_class[cls] += len(gts)

    # --- Matching and Error Calculation Loop ---
    for sample_token in set(pred_by_sample.keys()) | set(gt_by_sample.keys()):
        for cls in CLASS_THRESHOLDS:
            # Filter predictions by a confidence threshold
            preds = [p for p in pred_by_sample.get(sample_token, {}).get(cls, []) if p['detection_score'] > 0.20]
            gts = gt_by_sample.get(sample_token, {}).get(cls, [])
            matched_gt_indices = [False] * len(gts)
            
            # Sort predictions by score to prioritize high-confidence matches
            preds.sort(key=lambda x: x['detection_score'], reverse=True)

            for pred in preds:
                pred_pos = np.array(pred['translation'])
                best_dist = float('inf')
                best_gt_idx = -1

                # Find the best available ground truth match
                for i, gt in enumerate(gts):
                    if matched_gt_indices[i]:
                        continue
                    dist = np.linalg.norm(pred_pos - np.array(gt['translation']))
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = i

                # Check if the match is within the class-specific distance threshold
                if best_gt_idx != -1 and best_dist < CLASS_THRESHOLDS.get(cls, 2.0):
                    matched_gt_indices[best_gt_idx] = True
                    class_matches[cls].append((pred['detection_score'], True)) # True Positive
                    
                    gt_match = gts[best_gt_idx]
                    errors[cls]['ate'].append(best_dist)
                    errors[cls]['ase'].append(1.0 - aligned_iou(pred['size'], gt_match['size']))
                    if 'rotation' in pred and 'rotation' in gt_match:
                         q1, q2 = np.array(pred['rotation']), np.array(gt_match['rotation'])
                         errors[cls]['aoe'].append(quaternion_yaw_angle_diff(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)))
                    if 'velocity' in pred and 'velocity' in gt_match:
                        errors[cls]['ave'].append(np.linalg.norm(np.array(pred['velocity']) - np.array(gt_match['velocity'])))
                    if 'attribute_name' in pred and 'attribute_name' in gt_match:
                        errors[cls]['aae'].append(float(pred['attribute_name'] != gt_match['attribute_name']))
                else:
                    class_matches[cls].append((pred['detection_score'], False)) # False Positive

    # --- Final Metrics Calculation ---
    print("\n--- Class-wise Metrics ---")
    print(f"{'Class':<20} {'TP':>5} {'FP':>5} {'FN':>5} | {'AP':>5} | {'ATE':>5} {'ASE':>5} {'AOE':>5} {'AVE':>5} {'AAE':>5}")
    print("-" * 85)

    final_metrics = {}
    total_ap = 0
    ap_valid_classes = 0

    for cls in sorted(CLASS_THRESHOLDS.keys()):
        ap = calculate_ap(class_matches[cls], total_gts_per_class[cls])
        final_metrics[cls] = {'AP': ap}
        
        tp = sum(1 for _, is_tp in class_matches[cls] if is_tp)
        fp = len(class_matches[cls]) - tp
        fn = total_gts_per_class[cls] - tp
        
        if not np.isnan(ap):
            total_ap += ap
            ap_valid_classes += 1

        metric_names = ['ate', 'ase', 'aoe', 'ave', 'aae']
        for m in metric_names:
            final_metrics[cls][m.upper()] = np.mean(errors[cls][m]) if errors[cls][m] else float('nan')

        print(f"{cls:<20} {tp:>5} {fp:>5} {fn:>5} | {final_metrics[cls]['AP']:5.3f} | "
              f"{final_metrics[cls]['ATE']:5.2f} {final_metrics[cls]['ASE']:5.2f} "
              f"{final_metrics[cls]['AOE']:5.2f} {final_metrics[cls]['AVE']:5.2f} "
              f"{final_metrics[cls]['AAE']:5.2f}")

    mAP = total_ap / ap_valid_classes if ap_valid_classes > 0 else 0.0

    # --- NDS Calculation ---
    mATE = np.nanmean([1 - final_metrics[cls]['ATE'] for cls in CLASS_THRESHOLDS if not np.isnan(final_metrics[cls]['ATE'])])
    mASE = np.nanmean([1 - final_metrics[cls]['ASE'] for cls in CLASS_THRESHOLDS if not np.isnan(final_metrics[cls]['ASE'])])
    mAOE = np.nanmean([1 - final_metrics[cls]['AOE'] for cls in CLASS_THRESHOLDS if not np.isnan(final_metrics[cls]['AOE'])])
    mAVE = np.nanmean([1 - final_metrics[cls]['AVE'] for cls in CLASS_THRESHOLDS if not np.isnan(final_metrics[cls]['AVE'])])
    mAAE = np.nanmean([1 - final_metrics[cls]['AAE'] for cls in CLASS_THRESHOLDS if not np.isnan(final_metrics[cls]['AAE'])])

    NDS = 0.4 * mAP + 0.1 * mATE + 0.1 * mASE + 0.1 * mAOE + 0.2 * mAVE + 0.1 * mAAE

    print("\n--- Overall Scores ---")
    print(f"mAP: {mAP:.4f}")
    print(f"NDS: {NDS:.4f}")

# --- Main Execution ---
if __name__ == "__main__":
    path_to_predictions = './submission/pts_bbox/results_nusc_nofinetuning.json'
    path_to_ground_truth = './results_nusc_with_gt.json'
    
    pred_data = load_results(path_to_predictions)
    gt_data = load_results(path_to_ground_truth)

    if pred_data and gt_data:
        evaluate(pred_data, gt_data)
    else:
        print("\nEvaluation skipped due to file loading errors.")

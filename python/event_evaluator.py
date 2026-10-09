import numpy as np

def evaluate_event_metrics(
    t_frames: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    tolerance_sec: float = 0.001,     # ±1 ms window for valid transition match
    search_window_sec: float = 0.005   # 5 ms max search window for latency
) -> dict:
    """
    Evaluates state-transition performance using event-matching logic.
    
    Parameters:
        t_frames          : Array of frame timestamps (seconds)
        gt_labels         : Ground-truth state sequence (signed integers)
        pred_labels       : Predicted state sequence from HMM/tracker
        tolerance_sec     : Match window around true transition for FAR check
        search_window_sec : Max allowed latency window before marking as missed
        
    Returns:
        results           : Dictionary containing FAR (events/s), mean latency (ms),
                            missed transitions, and total runtime duration.
    """
    total_duration = t_frames[-1] - t_frames[0]
    
    # 1. Identify discrete transition event indices (where state changes)
    gt_change_indices = np.where(gt_labels[1:] != gt_labels[:-1])[0] + 1
    pred_change_indices = np.where(pred_labels[1:] != pred_labels[:-1])[0] + 1
    
    gt_events = [(t_frames[idx], gt_labels[idx]) for idx in gt_change_indices]
    pred_events = [(t_frames[idx], pred_labels[idx]) for idx in pred_change_indices]
    
    matched_pred_indices = set()
    latencies = []
    missed_gt_count = 0
    
    # 2. Match predicted transitions against ground-truth transitions
    for t_gt, state_gt in gt_events:
        matched = False
        
        for p_i, (t_pred, state_pred) in enumerate(pred_events):
            if p_i in matched_pred_indices:
                continue
                
            # Check if prediction occurs after GT transition within search window
            if 0.0 <= (t_pred - t_gt) <= search_window_sec:
                if state_pred == state_gt:
                    latency = t_pred - t_gt
                    latencies.append(latency)
                    matched_pred_indices.add(p_i)
                    matched = True
                    break
                    
        if not matched:
            missed_gt_count += 1
            
    # 3. Classify unmatched predictions as False Alarms
    false_alarm_count = 0
    for p_i, (t_pred, state_pred) in enumerate(pred_events):
        if p_i in matched_pred_indices:
            continue
            
        # Check if predicted transition sits within tolerance of ANY GT event
        near_gt = any(abs(t_pred - t_gt) <= tolerance_sec for t_gt, _ in gt_events)
        if not near_gt:
            false_alarm_count += 1

    # 4. Compute final metric scores
    far_per_sec = false_alarm_count / total_duration if total_duration > 0 else 0.0
    mean_latency_ms = (np.mean(latencies) * 1000.0) if len(latencies) > 0 else np.nan
    max_latency_ms = (np.max(latencies) * 1000.0) if len(latencies) > 0 else np.nan
    
    return {
        "Total Test Duration (s)": round(total_duration, 4),
        "True Transition Events": len(gt_events),
        "Matched Transitions": len(latencies),
        "Missed Transitions": missed_gt_count,
        "False Alarm Events": false_alarm_count,
        "FAR (events/s)": round(far_per_sec, 3),
        "Mean Latency (ms)": round(mean_latency_ms, 3) if not np.isnan(mean_latency_ms) else "N/A",
        "Max Latency (ms)": round(max_latency_ms, 3) if not np.isnan(max_latency_ms) else "N/A"
    }
import numpy as np

def calculate_accuracy(truth, estimate):
    """Percentage of frames where the estimated state matches ground truth."""
    return np.mean(truth == estimate) * 100

def calculate_far(truth, estimate):
    """False Alarm Rate: % of frames showing a transition when none occurred."""
    truth_transitions = np.diff(truth) != 0
    est_transitions = np.diff(estimate) != 0
    
    # False alarm: Estimate transitioned, but truth stayed the same
    false_alarms = np.sum(est_transitions & ~truth_transitions)
    total_stable_frames = np.sum(~truth_transitions)
    
    if total_stable_frames == 0:
        return 0.0
    return (false_alarms / total_stable_frames) * 100

def calculate_latency(truth, estimate, frame_times):
    """Average latency in microseconds to detect a real transition."""
    truth_transitions = np.where(np.diff(truth) != 0)[0]
    latencies = []
    
    for t_idx in truth_transitions:
        # Look ahead in the estimate array for the transition
        window = estimate[t_idx : t_idx + 50] # Check next 50 frames
        est_transition_idx = np.where(np.diff(window) != 0)[0]
        
        if len(est_transition_idx) > 0:
            # Calculate time difference between truth and estimate
            time_diff = frame_times[t_idx + est_transition_idx[0]] - frame_times[t_idx]
            latencies.append(time_diff * 1e6) # Convert to microseconds
            
    return np.mean(latencies) if latencies else float('inf')
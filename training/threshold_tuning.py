import numpy as np
from sklearn.metrics import f1_score


def tune_threshold(predictions: np.ndarray, labels: np.ndarray, 
                   search_range=(0.0, 1.0), step=0.005):
    if len(predictions) == 0 or len(labels) == 0:
        return 0.5
    
    if labels.sum() == 0:
        return 0.5
    
    thresholds = np.arange(search_range[0], search_range[1] + step, step)
    scores = []
    
    for th in thresholds:
        pred_labels = (predictions >= th).astype(int)
        f1 = f1_score(labels, pred_labels, zero_division=0)
        scores.append(f1)
    
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = scores[best_idx]
    
    return optimal_threshold, best_f1


def tune_threshold_with_grid(predictions: np.ndarray, labels: np.ndarray):
    if len(predictions) == 0 or len(labels) == 0 or labels.sum() == 0:
        return 0.5
    
    # Stage 1: Coarse search
    coarse_thresholds = np.arange(0.0, 1.01, 0.05)
    coarse_scores = [
        f1_score(labels, (predictions >= th).astype(int), zero_division=0)
        for th in coarse_thresholds
    ]
    best_coarse_idx = np.argmax(coarse_scores)
    best_coarse_th = coarse_thresholds[best_coarse_idx]
    
    # Stage 2: Fine search around best coarse threshold
    lower = max(0.0, best_coarse_th - 0.05)
    upper = min(1.0, best_coarse_th + 0.05)
    fine_thresholds = np.arange(lower, upper + 0.001, 0.001)
    fine_scores = [
        f1_score(labels, (predictions >= th).astype(int), zero_division=0)
        for th in fine_thresholds
    ]
    
    best_fine_idx = np.argmax(fine_scores)
    optimal_threshold = fine_thresholds[best_fine_idx]
    best_f1 = fine_scores[best_fine_idx]
    
    return optimal_threshold, best_f1


def apply_threshold(predictions: np.ndarray, threshold: float):
    return (predictions >= threshold).astype(np.int8)


def evaluate_threshold(predictions: np.ndarray, labels: np.ndarray, threshold: float):
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    pred_labels = apply_threshold(predictions, threshold)
    
    return {
        "threshold": threshold,
        "f1": f1_score(labels, pred_labels, zero_division=0),
        "precision": precision_score(labels, pred_labels, zero_division=0),
        "recall": recall_score(labels, pred_labels, zero_division=0),
        "accuracy": accuracy_score(labels, pred_labels),
    }


def analyze_threshold_curve(predictions: np.ndarray, labels: np.ndarray, 
                            n_points=100, plot=False):
    thresholds = np.linspace(0, 1, n_points)
    metrics = [evaluate_threshold(predictions, labels, th) for th in thresholds]
    
    f1_scores = [m["f1"] for m in metrics]
    best_idx = np.argmax(f1_scores)
    
    result = {
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "best_threshold": thresholds[best_idx],
        "best_f1": f1_scores[best_idx],
        "all_metrics": metrics,
    }
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)
            plt.axvline(result["best_threshold"], color='red', linestyle='--', 
                       label=f'Best Threshold = {result["best_threshold"]:.3f}')
            plt.axhline(result["best_f1"], color='green', linestyle='--', 
                       label=f'Best F1 = {result["best_f1"]:.3f}')
            plt.xlabel("Threshold")
            plt.ylabel("F1 Score")
            plt.title("F1 Score vs Threshold")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    return result


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.random(1000)
    
    print("Testing threshold tuning...")
    opt_th, best_f1 = tune_threshold(y_pred, y_true)
    print(f"Optimal threshold: {opt_th:.3f}")
    print(f"Best F1 score: {best_f1:.3f}")
    
    print("\nEvaluating threshold...")
    metrics = evaluate_threshold(y_pred, y_true, opt_th)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
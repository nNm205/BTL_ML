import sys
import traceback
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict

sys.path.append("/kaggle/usr/lib/mabe-f-beta")
try:
    from metric import score as official_score
    OFFICIAL_METRIC_AVAILABLE = True
except:
    OFFICIAL_METRIC_AVAILABLE = False
    print("⚠️ Official metric not available, using custom implementation")


from config.config import INPUT_DIR, TRAIN_ANNOTATION_DIR


def compute_validation_metrics(submission: pd.DataFrame, verbose: bool = True):
    try:
        # Load solution
        dataset = pl.read_csv(INPUT_DIR / "train.csv").to_pandas()
        
        solution = []
        for _, row in dataset.iterrows():
            lab_id = row["lab_id"]
            
            # Skip MABe22 labs
            if lab_id.startswith("MABe22"):
                continue
            
            video_id = row["video_id"]
            path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
            
            try:
                annot = pd.read_parquet(path)
            except FileNotFoundError:
                continue
            
            annot["lab_id"] = lab_id
            annot["video_id"] = video_id
            annot["behaviors_labeled"] = row["behaviors_labeled"]
            
            # Format IDs
            annot["target_id"] = np.where(
                annot.target_id != annot.agent_id,
                annot["target_id"].apply(lambda s: f"mouse{s}"),
                "self"
            )
            annot["agent_id"] = annot["agent_id"].apply(lambda s: f"mouse{s}")
            
            solution.append(annot)
        
        solution = pd.concat(solution)
        
        # Filter solution to match submission videos
        solution_videos = set(submission["video_id"].unique())
        solution = solution[solution["video_id"].isin(solution_videos)]
        
        if len(solution) == 0:
            if verbose:
                print("⚠️ No matching videos found in solution")
            return {}
        
        # Tách single và pair
        submission_single = submission[submission["target_id"] == "self"].copy()
        submission_pair = submission[submission["target_id"] != "self"].copy()
        
        metrics = {
            "total_predictions": len(submission),
            "single_predictions": len(submission_single),
            "pair_predictions": len(submission_pair),
        }
        
        # Tính F1 tổng thể (nếu có official metric)
        if OFFICIAL_METRIC_AVAILABLE:
            try:
                overall_f1 = official_score(solution, submission, "row_id", beta=1.0)
                metrics["overall_f1"] = overall_f1
            except Exception as e:
                if verbose:
                    print(f"⚠️ Cannot compute official F1: {e}")
        
        # Tính F1 per-action
        action_metrics = compute_per_action_metrics(solution, submission)
        metrics["per_action"] = action_metrics
        
        # Print summary
        if verbose:
            print_metrics_summary(metrics)
        
        return metrics
        
    except Exception as e:
        if verbose:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"\n⚠️ Warning: Could not compute validation metrics: {error_msg}")
            if verbose:
                print(f"Traceback: {traceback.format_exc()[:300]}")
        return {}


def compute_per_action_metrics(solution: pd.DataFrame, submission: pd.DataFrame):
    solution_pl = pl.DataFrame(solution)
    submission_pl = pl.DataFrame(submission)
    
    # Tạo keys
    solution_pl = solution_pl.with_columns(
        pl.concat_str([
            pl.col("video_id").cast(pl.Utf8),
            pl.col("agent_id").cast(pl.Utf8),
            pl.col("target_id").cast(pl.Utf8),
            pl.col("action"),
        ], separator="_").alias("label_key")
    )
    
    submission_pl = submission_pl.with_columns(
        pl.concat_str([
            pl.col("video_id").cast(pl.Utf8),
            pl.col("agent_id").cast(pl.Utf8),
            pl.col("target_id").cast(pl.Utf8),
            pl.col("action"),
        ], separator="_").alias("prediction_key")
    )
    
    action_stats = defaultdict(lambda: {
        "single": {"count": 0, "f1": 0.0},
        "pair": {"count": 0, "f1": 0.0}
    })
    
    # Tính F1 cho từng lab
    for lab in solution_pl["lab_id"].unique():
        lab_solution = solution_pl.filter(pl.col("lab_id") == lab).clone()
        lab_videos = set(lab_solution["video_id"].unique())
        lab_submission = submission_pl.filter(pl.col("video_id").is_in(lab_videos)).clone()
        
        # Build frame sets
        label_frames = defaultdict(set)
        prediction_frames = defaultdict(set)
        
        for row in lab_solution.to_dicts():
            label_frames[row["label_key"]].update(
                range(row["start_frame"], row["stop_frame"])
            )
        
        for row in lab_submission.to_dicts():
            key = row["prediction_key"]
            prediction_frames[key].update(
                range(row["start_frame"], row["stop_frame"])
            )
        
        # Compute F1 for each key
        all_keys = set(list(label_frames.keys()) + list(prediction_frames.keys()))
        
        for key in all_keys:
            action = key.split("_")[-1]
            mode = "single" if "self" in key else "pair"
            
            pred_frames = prediction_frames.get(key, set())
            label_frames_set = label_frames.get(key, set())
            
            tp = len(pred_frames & label_frames_set)
            fn = len(label_frames_set - pred_frames)
            fp = len(pred_frames - label_frames_set)
            
            if tp + fn + fp > 0:
                f1 = (1 + 1**2) * tp / ((1 + 1**2) * tp + 1**2 * fn + fp)
                action_stats[action][mode]["count"] += 1
                action_stats[action][mode]["f1"] += f1
    
    return dict(action_stats)


def print_metrics_summary(metrics: dict):
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    
    if "overall_f1" in metrics:
        print(f"Overall F1 Score: {metrics['overall_f1']:.4f}")
    
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"  - Single behaviors: {metrics['single_predictions']}")
    print(f"  - Pair behaviors: {metrics['pair_predictions']}")
    
    if "per_action" in metrics:
        print("\nPer-Action Performance:")
        print(f"{'-'*60}")
        print(f"{'Action':<20} {'Mode':<10} {'Count':<10} {'Avg F1':<10}")
        print(f"{'-'*60}")
        
        for action in sorted(metrics["per_action"].keys()):
            for mode in ["single", "pair"]:
                stats = metrics["per_action"][action][mode]
                if stats["count"] > 0:
                    avg_f1 = stats["f1"] / stats["count"]
                    print(f"{action:<20} {mode:<10} {stats['count']:<10} {avg_f1:<10.4f}")
        
        # Summary by mode
        single_actions = [
            a for a in metrics["per_action"].keys()
            if metrics["per_action"][a]["single"]["count"] > 0
        ]
        pair_actions = [
            a for a in metrics["per_action"].keys()
            if metrics["per_action"][a]["pair"]["count"] > 0
        ]
        
        if single_actions:
            single_avg = np.mean([
                metrics["per_action"][a]["single"]["f1"] / metrics["per_action"][a]["single"]["count"]
                for a in single_actions
            ])
            print(f"\nSingle behaviors: {len(single_actions)} actions, Avg F1: {single_avg:.4f}")
        
        if pair_actions:
            pair_avg = np.mean([
                metrics["per_action"][a]["pair"]["f1"] / metrics["per_action"][a]["pair"]["count"]
                for a in pair_actions
            ])
            print(f"Pair behaviors: {len(pair_actions)} actions, Avg F1: {pair_avg:.4f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("✓ Metrics module loaded")
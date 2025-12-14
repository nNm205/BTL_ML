import gc
import re
import sys
import time
import datetime
import argparse
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from config.config import (
    WORKING_DIR, RESULTS_DIR, 
    SELF_FEATURES_DIR, PAIR_FEATURES_DIR,
    TRAIN_ANNOTATION_DIR, INDEX_COLS
)
from preprocessing.data_loader import (
    load_train_dataframe,
    parse_behaviors_labeled,
    split_self_pair_behaviors,
    get_label_frames
)
from training.trainer import BehaviorModelTrainer


def train_self_behaviors(behavior_df: pl.DataFrame):
    """Train models cho self behaviors"""
    groups = behavior_df.group_by("lab_id", "behavior", maintain_order=True)
    total_groups = len(list(groups))
    start_time = time.perf_counter()
    
    results = []
    
    print(f"\n{'='*70}")
    print("TRAINING SELF BEHAVIORS")
    print(f"{'='*70}")
    print(f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'TIME':^15}|")
    print(f"{'-'*70}")
    
    for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
        index_list = []
        feature_list = []
        label_list = []
        
        for row in group.rows(named=True):
            video_id = row["video_id"]
            agent = row["agent"]
            agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
            
            # Load features
            feature_path = SELF_FEATURES_DIR / f"{video_id}.parquet"
            if not feature_path.exists():
                continue
            
            data = pl.scan_parquet(feature_path).filter(
                pl.col("agent_mouse_id") == agent_mouse_id
            )
            
            index = data.select(INDEX_COLS).collect(engine="streaming").with_columns([
                pl.col("video_id").cast(pl.Int32),
                pl.col("video_frame").cast(pl.Int32),
                pl.col("agent_mouse_id").cast(pl.Int32),
                pl.col("target_mouse_id").cast(pl.Int32)
            ])
            
            feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")
            
            # Load annotation
            annotation_path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
            if annotation_path.exists():
                annotation = (
                    pl.scan_parquet(annotation_path)
                    .filter(
                        (pl.col("action") == behavior) & 
                        (pl.col("agent_id") == agent_mouse_id)
                    )
                    .collect()
                )
            else:
                annotation = pl.DataFrame(schema={
                    "agent_id": pl.Int8, "target_id": pl.Int8,
                    "action": str, "start_frame": pl.Int16, "stop_frame": pl.Int16
                })
            
            # Get label frames
            label_frames = set()
            for ann_row in annotation.rows(named=True):
                label_frames.update(range(ann_row["start_frame"], ann_row["stop_frame"]))
            
            label = index.select(
                pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label")
            )
            
            if label.get_column("label").sum() == 0:
                continue
            
            index_list.append(index)
            feature_list.append(feature)
            label_list.append(label.get_column("label"))
        
        # Skip nếu không có data
        if not index_list:
            elapsed = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
            tqdm.write(f"|{lab_id:^25}|{behavior:^15}|{0:>10}|{0:>10}|{0:>10}|{'-':>10}|{str(elapsed):>15}|")
            continue
        
        # Concat data
        indices = pl.concat(index_list, how="vertical")
        features = pl.concat(feature_list, how="vertical")
        labels = pl.concat(label_list, how="vertical")
        
        del index_list, feature_list, label_list
        gc.collect()
        
        # Train
        result_dir = RESULTS_DIR / lab_id / behavior
        trainer = BehaviorModelTrainer(lab_id, behavior, result_dir)
        
        tqdm.write(f"|{lab_id:^25}|{behavior:^15}|{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")
        
        f1 = trainer.train(features, labels, indices)
        
        elapsed = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{f1:>10.2f}|{str(elapsed):>15}|")
        
        results.append({
            "lab_id": lab_id,
            "behavior": behavior,
            "type": "self",
            "f1": f1,
            "n_samples": len(indices),
            "n_positive": int(labels.sum()),
        })
        
        gc.collect()
    
    return results


def train_pair_behaviors(behavior_df: pl.DataFrame):
    """Train models cho pair behaviors"""
    groups = behavior_df.group_by("lab_id", "behavior", maintain_order=True)
    total_groups = len(list(groups))
    start_time = time.perf_counter()
    
    results = []
    
    print(f"\n{'='*70}")
    print("TRAINING PAIR BEHAVIORS")
    print(f"{'='*70}")
    print(f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'TIME':^15}|")
    print(f"{'-'*70}")
    
    for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
        index_list = []
        feature_list = []
        label_list = []
        
        for row in group.rows(named=True):
            video_id = row["video_id"]
            agent = row["agent"]
            target = row["target"]
            
            agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
            target_mouse_id = int(re.search(r"mouse(\d+)", target).group(1))
            
            # Load features
            feature_path = PAIR_FEATURES_DIR / f"{video_id}.parquet"
            if not feature_path.exists():
                continue
            
            data = pl.scan_parquet(feature_path).filter(
                (pl.col("agent_mouse_id") == agent_mouse_id) &
                (pl.col("target_mouse_id") == target_mouse_id)
            )
            
            index = data.select(INDEX_COLS).collect(engine="streaming")
            feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")
            
            # Load annotation
            annotation_path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
            if annotation_path.exists():
                annotation = (
                    pl.scan_parquet(annotation_path)
                    .filter(
                        (pl.col("action") == behavior) &
                        (pl.col("agent_id") == agent_mouse_id) &
                        (pl.col("target_id") == target_mouse_id)
                    )
                    .collect()
                )
            else:
                annotation = pl.DataFrame(schema={
                    "agent_id": pl.Int8, "target_id": pl.Int8,
                    "action": str, "start_frame": pl.Int16, "stop_frame": pl.Int16
                })
            
            # Get label frames
            label_frames = set()
            for ann_row in annotation.rows(named=True):
                label_frames.update(range(ann_row["start_frame"], ann_row["stop_frame"]))
            
            label = index.select(
                pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label")
            )
            
            if label.get_column("label").sum() == 0:
                continue
            
            index_list.append(index)
            feature_list.append(feature)
            label_list.append(label.get_column("label"))
        
        if not index_list:
            elapsed = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
            tqdm.write(f"|{lab_id:^25}|{behavior:^15}|{0:>10}|{0:>10}|{0:>10}|{'-':>10}|{str(elapsed):>15}|")
            continue
        
        indices = pl.concat(index_list, how="vertical")
        features = pl.concat(feature_list, how="vertical")
        labels = pl.concat(label_list, how="vertical")
        
        del index_list, feature_list, label_list
        gc.collect()
        
        result_dir = RESULTS_DIR / lab_id / behavior
        trainer = BehaviorModelTrainer(lab_id, behavior, result_dir)
        
        tqdm.write(f"|{lab_id:^25}|{behavior:^15}|{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")
        
        f1 = trainer.train(features, labels, indices)
        
        elapsed = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{f1:>10.2f}|{str(elapsed):>15}|")
        
        results.append({
            "lab_id": lab_id,
            "behavior": behavior,
            "type": "pair",
            "f1": f1,
            "n_samples": len(indices),
            "n_positive": int(labels.sum()),
        })
        
        gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models")
    parser.add_argument(
        "--behavior_type",
        choices=["self", "pair", "both"],
        default="both",
        help="Loại behavior cần train"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("SCRIPT 2: MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Behavior type: {args.behavior_type}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading training data...")
    train_df = load_train_dataframe()
    behavior_df = parse_behaviors_labeled(train_df)
    self_df, pair_df = split_self_pair_behaviors(behavior_df)
    
    print(f"✓ Self behaviors: {len(self_df)}")
    print(f"✓ Pair behaviors: {len(pair_df)}")
    
    all_results = []
    
    # Train
    if args.behavior_type in ["self", "both"]:
        self_results = train_self_behaviors(self_df)
        all_results.extend(self_results)
    
    if args.behavior_type in ["pair", "both"]:
        pair_results = train_pair_behaviors(pair_df)
        all_results.extend(pair_results)
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total models trained: {len(all_results)}")
    if all_results:
        avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)
        print(f"Average F1 score: {avg_f1:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
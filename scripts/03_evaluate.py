import re
import sys
import argparse
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
from config.config import WORKING_DIR, RESULTS_DIR, INDEX_COLS
from preprocessing.data_loader import (
    load_train_dataframe,
    parse_behaviors_labeled
)
from evaluation.robustify import robustify_submission, validate_submission_format
from evaluation.metrics import compute_validation_metrics, print_metrics_summary


def generate_oof_predictions(train_df: pl.DataFrame, behavior_df: pl.DataFrame):
    print("\n" + "="*70)
    print("GENERATING OOF PREDICTIONS")
    print("="*70 + "\n")
    
    group_oof_predictions = []
    groups = behavior_df.group_by("lab_id", "video_id", "agent", "target", maintain_order=True)
    
    for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(list(groups)), desc="Processing groups"):
        agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
        target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))
        
        prediction_dataframe_list = []
        
        for row in group.rows(named=True):
            behavior = row["behavior"]
            oof_path = RESULTS_DIR / lab_id / behavior / "oof_predictions.parquet"
            
            if not oof_path.exists():
                continue
            
            prediction = (
                pl.scan_parquet(oof_path)
                .filter(
                    (pl.col("video_id") == video_id) &
                    (pl.col("agent_mouse_id") == agent_mouse_id) &
                    (pl.col("target_mouse_id") == target_mouse_id)
                )
                .select(
                    *INDEX_COLS,
                    (pl.col("prediction") * pl.col("predicted_label")).alias(behavior)
                )
                .collect()
            )
            
            if len(prediction) == 0:
                continue
            
            prediction_dataframe_list.append(prediction)
        
        if not prediction_dataframe_list:
            continue
        
        # Concat predictions
        prediction_dataframe = pl.concat(prediction_dataframe_list, how="align")
        
        # Chọn behavior có score cao nhất
        cols = prediction_dataframe.select(pl.exclude(INDEX_COLS)).columns
        
        prediction_labels_dataframe = prediction_dataframe.with_columns(
            pl.struct(pl.exclude(INDEX_COLS))
            .map_elements(
                lambda row: "none" if sum(row.values()) == 0 
                           else cols[np.argmax(list(row.values()))],
                return_dtype=pl.String,
            )
            .alias("prediction")
        ).select(INDEX_COLS + ["prediction"])
        
        # Gom thành intervals
        group_oof_prediction = (
            prediction_labels_dataframe
            .filter(pl.col("prediction") != pl.col("prediction").shift(1))
            .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))
            .filter(pl.col("prediction") != "none")
            .select(
                pl.col("video_id"),
                ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),
                pl.when(pl.col("target_mouse_id") == -1)
                .then(pl.lit("self"))
                .otherwise("mouse" + pl.col("target_mouse_id").cast(str))
                .alias("target_id"),
                pl.col("prediction").alias("action"),
                pl.col("video_frame").alias("start_frame"),
                pl.col("stop_frame"),
            )
        )
        
        group_oof_predictions.append(group_oof_prediction)
    
    oof_predictions = pl.concat(group_oof_predictions, how="vertical")
    
    print(f"\n✓ Generated {len(oof_predictions)} prediction intervals")
    
    return oof_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--skip_robustify",
        action="store_true",
        help="Skip robustify step"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="oof_predictions.csv",
        help="Output filename"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("SCRIPT 3: MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Output file: {args.output}")
    print(f"Skip robustify: {args.skip_robustify}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading training data...")
    train_df = load_train_dataframe()
    behavior_df = parse_behaviors_labeled(train_df)
    
    # Generate OOF predictions
    oof_predictions = generate_oof_predictions(train_df, behavior_df)
    
    # Robustify (optional)
    if not args.skip_robustify:
        print("\nRobustifying predictions...")
        oof_predictions = robustify_submission(oof_predictions, train_df, train_test="train")
    
    # Validate format
    print("\nValidating submission format...")
    is_valid = validate_submission_format(oof_predictions)
    
    if not is_valid:
        print("⚠️ Warning: Submission format has issues")
    
    # Save
    output_path = WORKING_DIR / args.output
    oof_predictions.with_row_index("row_id").write_csv(output_path)
    print(f"\n✓ Saved predictions to: {output_path}")
    
    # Compute metrics
    print("\nComputing validation metrics...")
    oof_pandas = pd.read_csv(output_path)
    metrics = compute_validation_metrics(oof_pandas, verbose=True)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
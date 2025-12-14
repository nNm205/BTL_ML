import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import load_train_dataframe, get_video_metadata
from features.feature_engineering import (
    run_feature_engineering_parallel,
    run_feature_engineering_sequential,
    check_existing_features
)


def main():
    parser = argparse.ArgumentParser(description="Prepare features for training")
    parser.add_argument(
        "--mode",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Processing mode (parallel hoặc sequential)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Số CPU cores sử dụng (-1 = all)"
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Chỉ check features đã có, không tạo mới"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("SCRIPT 1: FEATURE PREPARATION")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    if args.mode == "parallel":
        print(f"Number of jobs: {args.n_jobs}")
    print(f"{'='*70}\n")
    
    # Check existing features
    print("Checking existing features...")
    self_count, pair_count = check_existing_features()
    
    if args.check_only:
        print("\n✓ Check complete. Exiting...")
        return
    
    # Load training data
    print("\nLoading training data...")
    train_df = load_train_dataframe()
    video_metadata = get_video_metadata(train_df)
    print(f"✓ Found {len(video_metadata)} videos with labeled behaviors")
    
    # Run feature engineering
    if args.mode == "sequential":
        successful = run_feature_engineering_sequential(video_metadata)
    else:
        successful = run_feature_engineering_parallel(
            video_metadata, 
            n_jobs=args.n_jobs
        )
    
    print(f"\n{'='*70}")
    print("FEATURE PREPARATION COMPLETED")
    print(f"Successfully processed: {len(successful)}/{len(video_metadata)} videos")
    print(f"{'='*70}\n")
    
    # Final check
    print("Final feature count:")
    check_existing_features()


if __name__ == "__main__":
    main()
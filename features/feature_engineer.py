import gc
import joblib
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm

from config.config import (
    TRAIN_TRACKING_DIR,
    SELF_FEATURES_DIR,
    PAIR_FEATURES_DIR,
)
from features.self_features import make_self_features
from features.pair_features import make_pair_features
from preprocessing.data_loader import load_tracking_data


def process_single_video(row: dict) -> int:
    lab_id = row["lab_id"]
    video_id = row["video_id"]
    
    try:
        # Load tracking data
        tracking = load_tracking_data(lab_id, video_id, TRAIN_TRACKING_DIR)
        
        # Tạo self features
        self_features = make_self_features(metadata=row, tracking=tracking)
        self_features.write_parquet(SELF_FEATURES_DIR / f"{video_id}.parquet")
        
        # Tạo pair features
        pair_features = make_pair_features(metadata=row, tracking=tracking)
        pair_features.write_parquet(PAIR_FEATURES_DIR / f"{video_id}.parquet")
        
        return video_id
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        return -1


def run_feature_engineering_parallel(video_metadata_list: list, n_jobs: int = -1):
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING PIPELINE")
    print(f"{'='*60}")
    print(f"Total videos to process: {len(video_metadata_list)}")
    print(f"Output directories:")
    print(f"  - Self features: {SELF_FEATURES_DIR}")
    print(f"  - Pair features: {PAIR_FEATURES_DIR}")
    print(f"{'='*60}\n")
    
    # Tạo thư mục output
    SELF_FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    PAIR_FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Chạy song song
    results = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(process_single_video)(row) 
        for row in video_metadata_list
    )
    
    # Filter ra các video thành công
    successful_videos = [vid for vid in results if vid != -1]
    
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING COMPLETED")
    print(f"Successfully processed: {len(successful_videos)}/{len(video_metadata_list)} videos")
    print(f"{'='*60}\n")
    
    gc.collect()
    
    return successful_videos


def run_feature_engineering_sequential(video_metadata_list: list):
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING PIPELINE (SEQUENTIAL)")
    print(f"{'='*60}")
    print(f"Total videos to process: {len(video_metadata_list)}")
    print(f"{'='*60}\n")
    
    SELF_FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    PAIR_FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    
    successful_videos = []
    
    for row in tqdm(video_metadata_list, desc="Processing videos"):
        video_id = process_single_video(row)
        if video_id != -1:
            successful_videos.append(video_id)
    
    print(f"\n{'='*60}")
    print(f"Successfully processed: {len(successful_videos)}/{len(video_metadata_list)} videos")
    print(f"{'='*60}\n")
    
    gc.collect()
    
    return successful_videos


def check_existing_features():
    """Kiểm tra xem đã có features nào được tạo chưa"""
    self_count = len(list(SELF_FEATURES_DIR.glob("*.parquet")))
    pair_count = len(list(PAIR_FEATURES_DIR.glob("*.parquet")))
    
    print(f"\nExisting features:")
    print(f"  - Self features: {self_count} files")
    print(f"  - Pair features: {pair_count} files")
    
    return self_count, pair_count


if __name__ == "__main__":
    from preprocessing.data_loader import load_train_dataframe, get_video_metadata
    
    print("Loading train data...")
    train_df = load_train_dataframe()
    video_metadata = get_video_metadata(train_df)
    
    print(f"Found {len(video_metadata)} videos with labeled behaviors")
    
    check_existing_features()
    
    # Chọn mode
    choice = input("\nRun parallel (p) or sequential (s)? [p/s]: ").strip().lower()
    
    if choice == 's':
        run_feature_engineering_sequential(video_metadata)
    else:
        run_feature_engineering_parallel(video_metadata, n_jobs=-1)
    
    check_existing_features()
    print("\n✓ Feature engineering completed!")
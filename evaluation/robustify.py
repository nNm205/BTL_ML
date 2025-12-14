import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from config.config import INPUT_DIR


def robustify_submission(submission: pl.DataFrame, dataset: pl.DataFrame, 
                        train_test: str = "train") -> pl.DataFrame:
    tracking_dir = INPUT_DIR / f"{train_test}_tracking"
    
    # ========================================================================
    # 1. Drop frames với start >= stop
    # ========================================================================
    old_len = len(submission)
    submission = submission.filter(pl.col("start_frame") < pl.col("stop_frame"))
    
    if len(submission) != old_len:
        print(f"⚠️ Dropped {old_len - len(submission)} rows with start_frame >= stop_frame")
    
    # ========================================================================
    # 2. Drop duplicate frames
    # ========================================================================
    old_len = len(submission)
    group_list = []
    
    for _, group in submission.group_by(["video_id", "agent_id", "target_id"]):
        group = group.sort("start_frame")
        
        # Tạo mask để loại bỏ các interval overlap
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        
        for i, row in enumerate(group.rows(named=True)):
            if row["start_frame"] < last_stop_frame:
                mask[i] = False  # Overlap -> loại bỏ
            else:
                last_stop_frame = row["stop_frame"]
        
        group_list.append(group.filter(pl.Series("mask", mask)))
    
    submission = pl.concat(group_list, how="vertical")
    
    if len(submission) != old_len:
        print(f"⚠️ Dropped {old_len - len(submission)} overlapping intervals")
    
    # ========================================================================
    # 3. Fill missing videos
    # ========================================================================
    existing_videos = set(submission.get_column("video_id").unique().to_list())
    all_videos = dataset.filter(pl.col("behaviors_labeled").is_not_null())
    
    missing_entries = []
    
    for row in all_videos.rows(named=True):
        lab_id = row["lab_id"]
        video_id = row["video_id"]
        
        # Skip nếu video này đã có predictions
        if video_id in existing_videos:
            continue
        
        # Skip nếu behaviors_labeled là string (corrupt data)
        if isinstance(row["behaviors_labeled"], str):
            continue
        
        print(f"⚠️ Video {video_id} has no predictions. Filling with dummy intervals...")
        
        # Load tracking để lấy frame range
        try:
            tracking_path = tracking_dir / f"{lab_id}/{video_id}.parquet"
            tracking = pd.read_parquet(tracking_path)
            start_frame = tracking.video_frame.min()
            stop_frame = tracking.video_frame.max() + 1
        except:
            print(f"   ⚠️ Cannot load tracking for video {video_id}, skipping...")
            continue
        
        # Parse behaviors_labeled
        try:
            behaviors = json.loads(row["behaviors_labeled"])
            behaviors = sorted(list({b.replace("'", "") for b in behaviors}))
            behaviors = [b.split(",") for b in behaviors]
            behaviors_df = pd.DataFrame(behaviors, columns=["agent", "target", "action"])
        except:
            print(f"   ⚠️ Cannot parse behaviors for video {video_id}, skipping...")
            continue
        
        # Tạo dummy intervals: chia đều video cho các behaviors
        for (agent, target), actions_group in behaviors_df.groupby(["agent", "target"]):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions_group)))
            
            for i, action_row in enumerate(actions_group.itertuples(index=False)):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                
                missing_entries.append({
                    "video_id": video_id,
                    "agent_id": agent,
                    "target_id": target,
                    "action": action_row.action,
                    "start_frame": batch_start,
                    "stop_frame": batch_stop,
                })
    
    # Concat missing entries
    if missing_entries:
        missing_df = pl.DataFrame(missing_entries)
        submission = pl.concat([submission, missing_df], how="vertical")
        print(f"✓ Filled {len(missing_entries)} missing intervals for {len(set(e['video_id'] for e in missing_entries))} videos")
    
    return submission


def validate_submission_format(submission: pl.DataFrame):
    required_cols = ["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
    
    # Check columns
    for col in required_cols:
        if col not in submission.columns:
            print(f"❌ Missing column: {col}")
            return False
    
    # Check start < stop
    invalid = submission.filter(pl.col("start_frame") >= pl.col("stop_frame"))
    if len(invalid) > 0:
        print(f"❌ Found {len(invalid)} rows with start_frame >= stop_frame")
        return False
    
    # Check for overlaps
    for (video_id, agent, target), group in submission.group_by(["video_id", "agent_id", "target_id"]):
        group = group.sort("start_frame")
        frames = group.select(["start_frame", "stop_frame"]).rows()
        
        for i in range(len(frames) - 1):
            if frames[i][1] > frames[i+1][0]:
                print(f"❌ Overlapping intervals found in video {video_id}, agent {agent}, target {target}")
                return False
    
    print("✓ Submission format is valid")
    return True


def merge_consecutive_same_action(submission: pl.DataFrame, max_gap: int = 0):
    result = []
    
    for (video_id, agent, target, action), group in submission.group_by(
        ["video_id", "agent_id", "target_id", "action"]
    ):
        group = group.sort("start_frame")
        
        merged = []
        current_start = None
        current_stop = None
        
        for row in group.rows(named=True):
            if current_start is None:
                current_start = row["start_frame"]
                current_stop = row["stop_frame"]
            else:
                # Nếu khoảng cách <= max_gap thì merge
                if row["start_frame"] - current_stop <= max_gap:
                    current_stop = max(current_stop, row["stop_frame"])
                else:
                    # Lưu interval hiện tại và bắt đầu interval mới
                    merged.append({
                        "video_id": video_id,
                        "agent_id": agent,
                        "target_id": target,
                        "action": action,
                        "start_frame": current_start,
                        "stop_frame": current_stop,
                    })
                    current_start = row["start_frame"]
                    current_stop = row["stop_frame"]
        
        # Lưu interval cuối cùng
        if current_start is not None:
            merged.append({
                "video_id": video_id,
                "agent_id": agent,
                "target_id": target,
                "action": action,
                "start_frame": current_start,
                "stop_frame": current_stop,
            })
        
        result.extend(merged)
    
    return pl.DataFrame(result)


if __name__ == "__main__":
    print("✓ Robustify module loaded")
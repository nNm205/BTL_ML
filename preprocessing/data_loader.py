import polars as pl
from pathlib import Path
from config.config import INPUT_DIR, SELF_BEHAVIORS, PAIR_BEHAVIORS


def load_train_dataframe():
    return pl.read_csv(INPUT_DIR / "train.csv")


def parse_behaviors_labeled(train_df: pl.DataFrame):
    behavior_df = (
        train_df
        .filter(pl.col("behaviors_labeled").is_not_null())
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled")
            .map_elements(eval, return_dtype=pl.List(pl.Utf8))
            .alias("behaviors_labeled_list"),
        )
        .explode("behaviors_labeled_list")
        .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled_element")
            .str.split(",")
            .list[0]
            .str.replace_all("'", "")
            .alias("agent"),
            pl.col("behaviors_labeled_element")
            .str.split(",")
            .list[1]
            .str.replace_all("'", "")
            .alias("target"),
            pl.col("behaviors_labeled_element")
            .str.split(",")
            .list[2]
            .str.replace_all("'", "")
            .alias("behavior"),
        )
    )
    return behavior_df


def split_self_pair_behaviors(behavior_df: pl.DataFrame):
    self_df = behavior_df.filter(pl.col("behavior").is_in(SELF_BEHAVIORS))
    pair_df = behavior_df.filter(pl.col("behavior").is_in(PAIR_BEHAVIORS))
    return self_df, pair_df


def get_video_metadata(train_df: pl.DataFrame):
    videos = (
        train_df
        .filter(pl.col("behaviors_labeled").is_not_null())
        .rows(named=True)
    )
    return list(videos)


def load_tracking_data(lab_id: str, video_id: int, data_dir: Path):
    path = data_dir / f"{lab_id}/{video_id}.parquet"
    return pl.read_parquet(path)


def load_annotation_data(lab_id: str, video_id: int, annotation_dir: Path):
    path = annotation_dir / lab_id / f"{video_id}.parquet"
    
    if not path.exists():
        return pl.DataFrame(
            schema={
                "agent_id": pl.Int8,
                "target_id": pl.Int8,
                "action": str,
                "start_frame": pl.Int16,
                "stop_frame": pl.Int16,
            }
        )
    
    return pl.read_parquet(path)


def get_label_frames(annotation_df: pl.DataFrame, behavior: str, agent_id: int, target_id: int = None):
    if target_id is None:
        filtered = annotation_df.filter(
            (pl.col("action") == behavior) & 
            (pl.col("agent_id") == agent_id)
        )
    else:
        filtered = annotation_df.filter(
            (pl.col("action") == behavior) & 
            (pl.col("agent_id") == agent_id) &
            (pl.col("target_id") == target_id)
        )
    
    label_frames = set()
    for row in filtered.rows(named=True):
        label_frames.update(range(row["start_frame"], row["stop_frame"]))
    
    return label_frames


if __name__ == "__main__":
    print("Loading train data...")
    train_df = load_train_dataframe()
    print(f"✓ Loaded {len(train_df)} videos")
    
    print("\nParsing behaviors...")
    behavior_df = parse_behaviors_labeled(train_df)
    print(f"✓ Parsed {len(behavior_df)} behavior instances")
    
    print("\nSplitting self/pair behaviors...")
    self_df, pair_df = split_self_pair_behaviors(behavior_df)
    print(f"✓ Self behaviors: {len(self_df)}")
    print(f"✓ Pair behaviors: {len(pair_df)}")
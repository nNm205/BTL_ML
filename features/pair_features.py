import itertools
import polars as pl
from config.config import BODY_PARTS, SPEED_PERIODS_MS
from utils.helpers import fill_missing_body_parts, calculate_facing_angle


def make_pair_features(metadata: dict, tracking: pl.DataFrame) -> pl.DataFrame:
    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        """Khoảng cách giữa 2 body parts (đơn vị: cm)"""
        return (
            (pl.col(f"{agent_or_target_1}_x_{body_part_1}") - 
             pl.col(f"{agent_or_target_2}_x_{body_part_2}")).pow(2) +
            (pl.col(f"{agent_or_target_1}_y_{body_part_1}") - 
             pl.col(f"{agent_or_target_2}_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]
    
    def body_part_speed(agent_or_target, body_part, period_ms):
        """Tốc độ của 1 body part trong khoảng thời gian period_ms (cm/s)"""
        window_frames = max(1, int(round(
            period_ms * metadata["frames_per_second"] / 1000.0
        )))
        
        speed_pixels = (
            (pl.col(f"{agent_or_target}_x_{body_part}").diff()).pow(2) +
            (pl.col(f"{agent_or_target}_y_{body_part}").diff()).pow(2)
        ).sqrt()
        
        speed_cm_per_frame = speed_pixels / metadata["pix_per_cm_approx"]
        speed_cm_per_sec = speed_cm_per_frame * metadata["frames_per_second"]
        
        return speed_cm_per_sec.rolling_mean(window_size=window_frames, center=True)
    
    def elongation(agent_or_target):
        """Elongation = body_length / head_width"""
        d1 = body_parts_distance(agent_or_target, "nose", agent_or_target, "tail_base")
        d2 = body_parts_distance(agent_or_target, "ear_left", agent_or_target, "ear_right")
        return d1 / (d2 + 1e-6)
    
    def body_angle(agent_or_target):
        """Góc uốn cơ thể (cosine similarity)"""
        v1x = pl.col(f"{agent_or_target}_x_nose") - pl.col(f"{agent_or_target}_x_body_center")
        v1y = pl.col(f"{agent_or_target}_y_nose") - pl.col(f"{agent_or_target}_y_body_center")
        v2x = pl.col(f"{agent_or_target}_x_tail_base") - pl.col(f"{agent_or_target}_x_body_center")
        v2y = pl.col(f"{agent_or_target}_y_tail_base") - pl.col(f"{agent_or_target}_y_body_center")
        
        dot = v1x * v2x + v1y * v2y
        mag = (v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt()
        
        return dot / (mag + 1e-6)
    
    def relative_position_angle(agent_prefix, target_prefix):
        """Góc vị trí tương đối: Agent ở góc nào so với hướng nhìn của Target?"""
        # Vector từ Target -> Agent
        rel_x = pl.col(f"{agent_prefix}_x_body_center") - pl.col(f"{target_prefix}_x_body_center")
        rel_y = pl.col(f"{agent_prefix}_y_body_center") - pl.col(f"{target_prefix}_y_body_center")
        
        # Vector hướng của Target
        target_dir_x = pl.col(f"{target_prefix}_x_nose") - pl.col(f"{target_prefix}_x_body_center")
        target_dir_y = pl.col(f"{target_prefix}_y_nose") - pl.col(f"{target_prefix}_y_body_center")
        
        # Cosine angle
        dot = (rel_x * target_dir_x) + (rel_y * target_dir_y)
        mag1 = (rel_x.pow(2) + rel_y.pow(2)).sqrt()
        mag2 = (target_dir_x.pow(2) + target_dir_y.pow(2)).sqrt()
        
        cos_angle = (dot / (mag1 * mag2 + 1e-6)).clip(-1.0, 1.0)
        return cos_angle.arccos().alias(f"{agent_prefix}_position_from_{target_prefix}_angle")
    
    def approach_speed():
        """Tốc độ tiếp cận/tránh xa (cm/s): âm = lại gần, dương = ra xa"""
        dist = body_parts_distance("agent", "body_center", "target", "body_center")
        return dist.diff().fill_null(0).alias("approach_speed")
    
    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================
    
    n_mice = sum([
        metadata["mouse1_strain"] is not None,
        metadata["mouse2_strain"] is not None,
        metadata["mouse3_strain"] is not None,
        metadata["mouse4_strain"] is not None
    ])
    
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()
    result = []
    
    # Pivot tracking
    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])
    
    pivot_trackings = {
        mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id)
        for mouse_id in range(1, n_mice + 1)
    }
    
    # Tạo features cho mỗi cặp (agent, target)
    for agent_mouse_id, target_mouse_id in itertools.permutations(range(1, n_mice + 1), 2):
        # Khởi tạo DataFrame kết quả cho cặp này
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": target_mouse_id,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )
        
        # Merge agent và target tracking
        merged_pivot = (
            pivot_trackings[agent_mouse_id]
            .select(pl.col("video_frame"), pl.exclude("video_frame").name.prefix("agent_"))
            .join(
                pivot_trackings[target_mouse_id].select(
                    pl.col("video_frame"),
                    pl.exclude("video_frame").name.prefix("target_"),
                ),
                on="video_frame",
                how="inner",
            )
        )
        
        # Fill missing body parts
        merged_pivot = fill_missing_body_parts(merged_pivot, BODY_PARTS, prefix="agent")
        merged_pivot = fill_missing_body_parts(merged_pivot, BODY_PARTS, prefix="target")
        
        # Tạo features
        features = merged_pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(target_mouse_id).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            
            # Facing angles
            calculate_facing_angle("agent", "target"),
            calculate_facing_angle("target", "agent"),
            
            # Relative position angles
            relative_position_angle("agent", "target"),
            relative_position_angle("target", "agent"),
            
            # Approach speed
            approach_speed(),
            
            # Distances giữa tất cả cặp body parts
            *[
                body_parts_distance("agent", agent_bp, "target", target_bp).alias(
                    f"at__{agent_bp}__{target_bp}__distance"
                )
                for agent_bp, target_bp in itertools.product(BODY_PARTS, repeat=2)
            ],
            
            # Speed của các body parts
            *[
                body_part_speed("agent", bp, period_ms).alias(
                    f"agent__{bp}__speed_{period_ms}ms"
                )
                for bp, period_ms in itertools.product(
                    ["ear_left", "ear_right", "tail_base"], 
                    SPEED_PERIODS_MS
                )
            ],
            *[
                body_part_speed("target", bp, period_ms).alias(
                    f"target__{bp}__speed_{period_ms}ms"
                )
                for bp, period_ms in itertools.product(
                    ["ear_left", "ear_right", "tail_base"], 
                    SPEED_PERIODS_MS
                )
            ],
            
            # Shape features
            elongation("agent").alias("agent__elongation"),
            elongation("target").alias("target__elongation"),
            body_angle("agent").alias("agent__body_angle"),
            body_angle("target").alias("target__body_angle"),
        )
        
        # Rolling aggregations
        features = features.with_columns(
            pl.col("at__nose__nose__distance").rolling_mean(30, center=True).alias("at__nose__nose__dist_mean_30"),
            pl.col("at__nose__tail_base__distance").rolling_mean(30, center=True).alias("at__nose__tail_base__dist_mean_30"),
            pl.col("agent_facing_target_angle").rolling_mean(15, center=True).alias("agent_facing_mean_15"),
            pl.col("target_facing_agent_angle").rolling_mean(15, center=True).alias("target_facing_mean_15"),
            
            # Approach speed rolling stats
            pl.col("approach_speed").rolling_mean(15, center=True).alias("approach_speed_mean_15"),
            pl.col("approach_speed").rolling_std(15, center=True).alias("approach_speed_std_15"),
            
            # Proximity duration
            (pl.col("at__nose__nose__distance") < 5.0).cast(pl.Float32).rolling_sum(30, center=True).alias("proximity_duration_30"),
            (pl.col("at__nose__nose__distance") < 10.0).cast(pl.Float32).rolling_sum(60, center=True).alias("proximity_duration_60"),
        )
        
        # Join với result_element
        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        
        result.append(result_element)
    
    return pl.concat(result, how="vertical")
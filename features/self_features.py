import polars as pl
import numpy as np
from config.config import BODY_PARTS, ROLLING_WINDOWS
from utils.helpers import (
    calculate_distance_expr,
    calculate_cosine_similarity,
    fill_missing_body_parts,
    create_rolling_features,
)


def make_self_features(metadata: dict, tracking: pl.DataFrame) -> pl.DataFrame:
    n_mice = 4
    
    pivot = tracking.pivot(
        on=["bodypart"], 
        index=["video_frame", "mouse_id"], 
        values=["x", "y"]
    ).sort(["mouse_id", "video_frame"])
    
    pivot_trackings = {
        m_id: pivot.filter(pl.col("mouse_id") == m_id) 
        for m_id in range(1, n_mice + 1)
    }
    
    result = []
    
    for agent_id in range(1, n_mice + 1):
        if agent_id not in pivot_trackings or pivot_trackings[agent_id].is_empty():
            continue
        
        agent_df = pivot_trackings[agent_id].select(
            pl.col("video_frame"),
            pl.exclude("video_frame").name.prefix("agent_")
        )
        
        agent_df = fill_missing_body_parts(agent_df, BODY_PARTS, prefix="agent")
        
        features = []
        
        # ====================================================================
        # PHẦN 1: GEOMETRY & SHAPE FEATURES
        # ====================================================================
        
        # 1.1 Body Length (nose -> tail_base)
        body_length = calculate_distance_expr(
            "agent_x_nose", "agent_y_nose", 
            "agent_x_tail_base", "agent_y_tail_base"
        )
        features.append(body_length.alias("agent_body_length"))
        
        # 1.2 Head Width (ear_left <-> ear_right)
        head_width = calculate_distance_expr(
            "agent_x_ear_left", "agent_y_ear_left",
            "agent_x_ear_right", "agent_y_ear_right"
        )
        features.append(head_width.alias("agent_head_width"))
        
        # 1.3 Elongation (body_length / head_width)
        features.append((body_length / (head_width + 1e-6)).alias("agent_elongation"))
        
        # 1.4 Body Curvature (góc uốn cong cơ thể)
        features.append(
            calculate_cosine_similarity(
                "agent_x_neck", "agent_y_neck",
                "agent_x_nose", "agent_y_nose",
                "agent_x_tail_base", "agent_y_tail_base"
            ).alias("agent_body_curvature")
        )
        
        # 1.5 Tail Curvature (quan trọng cho Rest/Huddle)
        features.append(
            calculate_cosine_similarity(
                "agent_x_tail_base", "agent_y_tail_base",
                "agent_x_body_center", "agent_y_body_center",
                "agent_x_tail_tip", "agent_y_tail_tip"
            ).alias("agent_tail_curvature")
        )
        
        # 1.6 Body Compactness
        body_width = calculate_distance_expr(
            "agent_x_lateral_left", "agent_y_lateral_left",
            "agent_x_lateral_right", "agent_y_lateral_right"
        )
        features.append((body_length / (body_width + 1e-6)).alias("agent_body_compactness"))
        
        # ====================================================================
        # PHẦN 2: DYNAMICS & MOTION FEATURES
        # ====================================================================
        
        # 2.1 Speed của body_center
        vel_x = pl.col("agent_x_body_center").diff().fill_null(0)
        vel_y = pl.col("agent_y_body_center").diff().fill_null(0)
        speed = (vel_x.pow(2) + vel_y.pow(2)).sqrt().alias("agent_speed")
        features.append(speed)
        
        # 2.2 Acceleration
        features.append(speed.diff().fill_null(0).alias("agent_acc"))
        
        # 2.3 Speed của nose
        vel_x_nose = pl.col("agent_x_nose").diff().fill_null(0)
        vel_y_nose = pl.col("agent_y_nose").diff().fill_null(0)
        speed_nose = (vel_x_nose.pow(2) + vel_y_nose.pow(2)).sqrt().alias("agent_nose_speed")
        features.append(speed_nose)
        
        # 2.4 Tail Speed (quan trọng cho Run vs Rest)
        vel_x_tail = pl.col("agent_x_tail_tip").diff().fill_null(0)
        vel_y_tail = pl.col("agent_y_tail_tip").diff().fill_null(0)
        speed_tail = (vel_x_tail.pow(2) + vel_y_tail.pow(2)).sqrt().alias("agent_tail_speed")
        features.append(speed_tail)
        
        # 2.5 Head Rotation Speed (cho Explore/Dig)
        head_angle = pl.arctan2(
            pl.col("agent_y_nose") - pl.col("agent_y_body_center"),
            pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        )
        head_rotation_speed = head_angle.diff().abs().fill_null(0).alias("agent_head_rotation_speed")
        features.append(head_rotation_speed)
        
        # 2.6 Vertical Movement (cho Rear/Climb)
        features.append(pl.col("agent_y_nose").alias("agent_nose_height"))
        features.append(
            pl.col("agent_y_nose").diff().fill_null(0).alias("agent_vertical_velocity")
        )
        
        # 2.7 Motion Consistency (Run = ổn định, Random walk = thay đổi)
        direction_x = vel_x / (speed + 1e-6)
        direction_y = vel_y / (speed + 1e-6)
        direction_change = (
            direction_x.diff().pow(2) + direction_y.diff().pow(2)
        ).sqrt().fill_null(0)
        features.append(direction_change.alias("agent_direction_change"))
        
        # ====================================================================
        # PHẦN 3: POSTURE FEATURES
        # ====================================================================
        
        # 3.1 Ear-to-Nose Distance (Rear: ears cao hơn nose)
        ear_left_to_nose = calculate_distance_expr(
            "agent_x_ear_left", "agent_y_ear_left",
            "agent_x_nose", "agent_y_nose"
        )
        ear_right_to_nose = calculate_distance_expr(
            "agent_x_ear_right", "agent_y_ear_right",
            "agent_x_nose", "agent_y_nose"
        )
        features.append(
            ((ear_left_to_nose + ear_right_to_nose) / 2).alias("agent_ear_nose_dist")
        )
        
        # 3.2 Body Angle (góc với horizon)
        body_dx = pl.col("agent_x_nose") - pl.col("agent_x_tail_base")
        body_dy = pl.col("agent_y_nose") - pl.col("agent_y_tail_base")
        body_angle = pl.arctan2(body_dy, body_dx).abs()
        features.append(body_angle.alias("agent_body_angle_rad"))
        
        # ====================================================================
        # PHẦN 4: ROLLING STATISTICS
        # ====================================================================
        
        # Các features cần rolling
        cols_to_roll = [
            speed,
            speed_nose,
            speed_tail,
            head_rotation_speed,
            body_length,
            direction_change
        ]
        
        for expr in cols_to_roll:
            col_name = expr.meta.output_name()
            
            # Rolling mean và std cho tất cả
            for w in ROLLING_WINDOWS:
                features.append(expr.rolling_mean(w, center=True).alias(f"{col_name}_mean_{w}"))
                features.append(expr.rolling_std(w, center=True).alias(f"{col_name}_std_{w}"))
                
                # Max cho speed và direction_change
                if "speed" in col_name or col_name == "agent_direction_change":
                    features.append(expr.rolling_max(w, center=True).alias(f"{col_name}_max_{w}"))
        
        # 4.1 Long-term Immobility Indicator (cho Rest/Freeze)
        for w in [60, 90, 120]:
            immobility_score = (speed < 0.5).cast(pl.Float32).rolling_mean(w, center=True)
            features.append(immobility_score.alias(f"agent_immobility_{w}"))
        
        # ====================================================================
        # COMBINE ALL FEATURES
        # ====================================================================
        
        final_cols = [
            "video_frame",
            pl.lit(agent_id).cast(pl.Int8).alias("agent_mouse_id"),
            pl.lit(-1).cast(pl.Int8).alias("target_mouse_id"),
        ]
        final_cols.extend(features)
        
        result.append(agent_df.select(final_cols))
    
    # Concat tất cả chuột
    if not result:
        return pl.DataFrame()
    
    return (
        pl.concat(result)
        .with_columns(pl.lit(metadata["video_id"]).cast(pl.Int32).alias("video_id"))
    )
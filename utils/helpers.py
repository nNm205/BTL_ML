import polars as pl
import numpy as np


def calculate_distance_expr(x1_col: str, y1_col: str, x2_col: str, y2_col: str):
    return (
        (pl.col(x1_col) - pl.col(x2_col)).pow(2) + 
        (pl.col(y1_col) - pl.col(y2_col)).pow(2)
    ).sqrt()


def calculate_cosine_similarity(x1: str, y1: str, x2: str, y2: str, x3: str, y3: str):
    ax = pl.col(x2) - pl.col(x1)
    ay = pl.col(y2) - pl.col(y1)
    bx = pl.col(x3) - pl.col(x1)
    by = pl.col(y3) - pl.col(y1)
    
    dot_product = (ax * bx) + (ay * by)
    mag_a = (ax.pow(2) + ay.pow(2)).sqrt()
    mag_b = (bx.pow(2) + by.pow(2)).sqrt()
    
    return dot_product / (mag_a * mag_b + 1e-6)


def calculate_angle_between_vectors(x1: str, y1: str, x2: str, y2: str, x3: str, y3: str):
    cos_sim = calculate_cosine_similarity(x1, y1, x2, y2, x3, y3)
    return cos_sim.clip(-1.0, 1.0).arccos()


def calculate_speed(x_col: str, y_col: str, fps: float = 30.0, pix_per_cm: float = 1.0):
    vel_x = pl.col(x_col).diff().fill_null(0)
    vel_y = pl.col(y_col).diff().fill_null(0)
    speed_pixels = (vel_x.pow(2) + vel_y.pow(2)).sqrt()
    return speed_pixels / pix_per_cm * fps


def calculate_acceleration(speed_expr: pl.Expr):
    return speed_expr.diff().fill_null(0)


def calculate_direction_vector(x_col: str, y_col: str):
    vel_x = pl.col(x_col).diff().fill_null(0)
    vel_y = pl.col(y_col).diff().fill_null(0)
    speed = (vel_x.pow(2) + vel_y.pow(2)).sqrt()
    
    dir_x = vel_x / (speed + 1e-6)
    dir_y = vel_y / (speed + 1e-6)
    
    return dir_x, dir_y


def calculate_direction_change(x_col: str, y_col: str):
    dir_x, dir_y = calculate_direction_vector(x_col, y_col)
    change = (dir_x.diff().pow(2) + dir_y.diff().pow(2)).sqrt().fill_null(0)
    return change


def calculate_body_angle(nose_x: str, nose_y: str, body_x: str, body_y: str, tail_x: str, tail_y: str):
    return calculate_cosine_similarity(body_x, body_y, nose_x, nose_y, tail_x, tail_y)


def calculate_facing_angle(agent_prefix: str, target_prefix: str):
    # Vector hướng nhìn của agent (body_center -> nose)
    v1_x = pl.col(f"{agent_prefix}_x_nose") - pl.col(f"{agent_prefix}_x_body_center")
    v1_y = pl.col(f"{agent_prefix}_y_nose") - pl.col(f"{agent_prefix}_y_body_center")
    
    # Vector từ agent đến target (body_center -> body_center)
    v2_x = pl.col(f"{target_prefix}_x_body_center") - pl.col(f"{agent_prefix}_x_body_center")
    v2_y = pl.col(f"{target_prefix}_y_body_center") - pl.col(f"{agent_prefix}_y_body_center")
    
    # Cosine của góc
    dot = (v1_x * v2_x) + (v1_y * v2_y)
    mag1 = (v1_x.pow(2) + v1_y.pow(2)).sqrt()
    mag2 = (v2_x.pow(2) + v2_y.pow(2)).sqrt()
    
    cos_angle = (dot / (mag1 * mag2 + 1e-6)).clip(-1.0, 1.0)
    return cos_angle.arccos().fill_null(-1.0)


def fill_missing_body_parts(df: pl.DataFrame, body_parts: list, prefix: str = "agent"):
    existing_cols = df.columns
    missing_cols = []
    
    for bp in body_parts:
        x_col = f"{prefix}_x_{bp}"
        y_col = f"{prefix}_y_{bp}"
        
        if x_col not in existing_cols:
            missing_cols.append(pl.lit(None).cast(pl.Float32).alias(x_col))
        if y_col not in existing_cols:
            missing_cols.append(pl.lit(None).cast(pl.Float32).alias(y_col))
    
    if missing_cols:
        df = df.with_columns(missing_cols)
    
    return df


def create_rolling_features(expr: pl.Expr, windows: list, stats: list = ["mean", "std", "max"]):
    features = []
    col_name = expr.meta.output_name()
    
    for w in windows:
        if "mean" in stats:
            features.append(expr.rolling_mean(w, center=True).alias(f"{col_name}_mean_{w}"))
        if "std" in stats:
            features.append(expr.rolling_std(w, center=True).alias(f"{col_name}_std_{w}"))
        if "max" in stats:
            features.append(expr.rolling_max(w, center=True).alias(f"{col_name}_max_{w}"))
        if "min" in stats:
            features.append(expr.rolling_min(w, center=True).alias(f"{col_name}_min_{w}"))
    
    return features
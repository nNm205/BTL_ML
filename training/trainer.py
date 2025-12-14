import gc
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from config.config import (
    N_FOLDS, RANDOM_STATE, XGB_MAX_BIN,
    RARE_BEHAVIOR_PARAMS, COMMON_BEHAVIOR_PARAMS,
    RARE_BEHAVIOR_ROUNDS, COMMON_BEHAVIOR_ROUNDS,
    RARE_EARLY_STOP, COMMON_EARLY_STOP,
    RARE_BEHAVIOR_THRESHOLD,
)
from training.threshold_tuning import tune_threshold


class BehaviorModelTrainer:
    def __init__(self, lab_id: str, behavior: str, result_dir: Path):
        self.lab_id = lab_id
        self.behavior = behavior
        self.result_dir = result_dir
        self.result_dir.mkdir(exist_ok=True, parents=True)
        
        self.is_rare = False
        self.models = [] 
        self.thresholds = [] 
        self.oof_predictions = None
        self.oof_labels = None
    
    def _determine_if_rare(self, labels: np.ndarray):
        pos_ratio = labels.sum() / len(labels)
        self.is_rare = pos_ratio < RARE_BEHAVIOR_THRESHOLD
        return self.is_rare
    
    def _get_hyperparameters(self, y_train: np.ndarray):
        # Calculate scale_pos_weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = min(n_neg / max(n_pos, 1), 100)
        
        # Chọn base params
        if self.is_rare:
            params = RARE_BEHAVIOR_PARAMS.copy()
            num_rounds = RARE_BEHAVIOR_ROUNDS
            early_stop = RARE_EARLY_STOP
        else:
            params = COMMON_BEHAVIOR_PARAMS.copy()
            num_rounds = COMMON_BEHAVIOR_ROUNDS
            early_stop = COMMON_EARLY_STOP
        
        params["scale_pos_weight"] = scale_pos_weight
        
        return params, num_rounds, early_stop
    
    def _train_single_fold_gpu(self, X_train, y_train, X_valid, y_valid, fold: int):
        params, num_rounds, early_stop = self._get_hyperparameters(y_train)
        
        params.update({
            "device": "cuda",
            "tree_method": "hist",
            "max_bin": XGB_MAX_BIN,
        })
        
        # QuantileDMatrix cho GPU
        dtrain = xgb.QuantileDMatrix(
            X_train, label=y_train, 
            feature_names=X_train.columns if hasattr(X_train, 'columns') else None,
            max_bin=XGB_MAX_BIN
        )
        dvalid = xgb.DMatrix(
            X_valid, label=y_valid,
            feature_names=X_valid.columns if hasattr(X_valid, 'columns') else None
        )
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=early_stop,
                    metric_name="logloss",
                    save_best=True
                )
            ],
            verbose_eval=0
        )
        
        predictions = model.predict(dvalid)
        
        del dtrain, dvalid
        gc.collect()
        
        return model, predictions
    
    def _train_single_fold_cpu(self, X_train, y_train, X_valid, y_valid, fold: int):
        params, num_rounds, early_stop = self._get_hyperparameters(y_train)
        
        params.update({
            "device": "cpu",
            "tree_method": "hist",
            "max_bin": 64,
        })
        
        dtrain = xgb.DMatrix(
            X_train, label=y_train,
            feature_names=X_train.columns if hasattr(X_train, 'columns') else None
        )
        dvalid = xgb.DMatrix(
            X_valid, label=y_valid,
            feature_names=X_valid.columns if hasattr(X_valid, 'columns') else None
        )
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=early_stop,
                    metric_name="logloss",
                    save_best=True
                )
            ],
            verbose_eval=0
        )
        
        predictions = model.predict(dvalid)
        
        del dtrain, dvalid
        gc.collect()
        
        return model, predictions
    
    def train(self, features, labels, indices):
        # Kiểm tra có sample dương không
        if labels.sum() == 0:
            with open(self.result_dir / "f1.txt", "w") as f:
                f.write("0.0\n")
            return 0.0
        
        # Xác định rare behavior
        self._determine_if_rare(labels)
        
        # Convert to numpy/pandas
        if hasattr(features, 'to_pandas'):
            X = features.to_pandas()
        else:
            X = features
        
        if hasattr(labels, 'to_pandas'):
            y = labels.to_pandas()
        else:
            y = labels
        
        if hasattr(indices, 'to_pandas'):
            groups = indices.to_pandas()["video_id"]
        else:
            groups = indices["video_id"]
        
        # Initialize OOF arrays
        n_samples = len(y)
        folds = np.ones(n_samples, dtype=np.int8) * -1
        oof_predictions = np.zeros(n_samples, dtype=np.float32)
        oof_prediction_labels = np.zeros(n_samples, dtype=np.int8)
        
        # K-Fold CV
        kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y, groups)):
            fold_dir = self.result_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            # Try GPU first
            try:
                model, fold_predictions = self._train_single_fold_gpu(
                    X_train, y_train, X_valid, y_valid, fold
                )
            except Exception as e:
                print(f"   ⚠️ GPU Failed for {self.behavior} fold {fold}: {e}")
                print("   >>> Falling back to CPU...")
                model, fold_predictions = self._train_single_fold_cpu(
                    X_train, y_train, X_valid, y_valid, fold
                )
            
            # Tune threshold
            try:
                threshold, _ = tune_threshold(fold_predictions, y_valid.values)
            except:
                threshold = 0.5
            
            # Save fold results
            folds[valid_idx] = fold
            oof_predictions[valid_idx] = fold_predictions
            oof_prediction_labels[valid_idx] = (fold_predictions >= threshold).astype(np.int8)
            
            self.models.append(model)
            self.thresholds.append(threshold)
            
            # Save model and threshold
            model.save_model(fold_dir / "model.json")
            with open(fold_dir / "threshold.txt", "w") as f:
                f.write(f"{threshold}\n")
            
            gc.collect()
        
        # Calculate final F1
        f1 = f1_score(y, oof_prediction_labels, zero_division=0)
        
        # Save results
        with open(self.result_dir / "f1.txt", "w") as f:
            f.write(f"{f1}\n")
        
        # Save OOF predictions
        import polars as pl
        oof_df = indices.with_columns(
            pl.Series("fold", folds, dtype=pl.Int8),
            pl.Series("prediction", oof_predictions, dtype=pl.Float32),
            pl.Series("predicted_label", oof_prediction_labels, dtype=pl.Int8),
        )
        oof_df.write_parquet(self.result_dir / "oof_predictions.parquet")
        
        self.oof_predictions = oof_predictions
        self.oof_labels = y
        
        return f1
    
    def predict(self, features, use_threshold=True):
        if not self.models:
            raise ValueError("No trained models found. Call train() first.")
        
        # Convert to pandas if needed
        if hasattr(features, 'to_pandas'):
            X = features.to_pandas()
        else:
            X = features
        
        # Average predictions từ các folds
        all_preds = []
        for model in self.models:
            dtest = xgb.DMatrix(X, feature_names=X.columns if hasattr(X, 'columns') else None)
            preds = model.predict(dtest)
            all_preds.append(preds)
        
        avg_preds = np.mean(all_preds, axis=0)
        
        if use_threshold:
            # Use average threshold
            avg_threshold = np.mean(self.thresholds)
            return (avg_preds >= avg_threshold).astype(np.int8)
        else:
            return avg_preds


if __name__ == "__main__":
    print("✓ Trainer module loaded")
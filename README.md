# Mouse Behavior Detection - BTL Machine Learning

**Sinh viên**: Nguyễn Nhật Minh  
**Mã sinh viên**: 23021631

**Sinh viên**: Trần Việt Hưng  
**Mã sinh viên**: 23021586

**Sinh viên**: Nguyễn Đình Quốc Huy  
**Mã sinh viên**: 23021574

## 📋 Mô tả dự án

Dự án phát hiện hành vi chuột sử dụng Machine Learning. Hệ thống phân tích video tracking data của chuột và dự đoán các hành vi như:

- **Self behaviors** (11 loại): selfgroom, rest, run, climb, dig, rear, huddle, freeze, etc.
- **Pair behaviors** (26 loại): sniff, chase, attack, mount, allogroom, etc.

## 🏗️ Cấu trúc dự án

```
mouse-behavior-detection/
│
├── config/
│   └── config.py                 # Cấu hình hệ thống
│
├── features/
│   ├── self_features.py          # Features cho hành vi cá nhân
│   ├── pair_features.py          # Features cho hành vi tương tác
│   └── feature_engineering.py    # Pipeline tạo features
│
├── preprocessing/
│   └── data_loader.py            # Load và parse dữ liệu
│
├── training/
│   ├── trainer.py                # XGBoost training logic
│   └── threshold_tuning.py       # Tối ưu threshold
│
├── evaluation/
│   ├── metrics.py                # Tính F1, validation metrics
│   └── robustify.py              # Post-processing predictions
│
├── utils/
│   └── helpers.py                # Các hàm tiện ích
│
├── scripts/
│   ├── 01_prepare_features.py    # Script 1: Tạo features
│   ├── 02_train_models.py        # Script 2: Train models
│   └── 03_evaluate.py            # Script 3: Đánh giá
│
├── requirements.txt
└── README.md
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/nNm205/BTL_ML
cd mouse-behavior-detection

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình đường dẫn

Mở file `config/config.py` và cập nhật đường dẫn data:

```python
INPUT_DIR = Path("path/to/your/data")
WORKING_DIR = Path("path/to/output")
```

### 3. Chạy pipeline

#### Bước 1: Tạo features

```bash
# Parallel mode (nhanh hơn)
python scripts/01_prepare_features.py --mode parallel --n_jobs -1

# Sequential mode (dùng khi RAM hạn chế)
python scripts/01_prepare_features.py --mode sequential

# Chỉ check features đã có
python scripts/01_prepare_features.py --check_only
```

#### Bước 2: Train models

```bash
# Train cả self và pair behaviors
python scripts/02_train_models.py --behavior_type both

# Chỉ train self behaviors
python scripts/02_train_models.py --behavior_type self

# Chỉ train pair behaviors
python scripts/02_train_models.py --behavior_type pair
```

#### Bước 3: Đánh giá kết quả

```bash
# Tạo OOF predictions và tính metrics
python scripts/03_evaluate.py --output oof_predictions.csv

# Skip robustify step
python scripts/03_evaluate.py --skip_robustify
```

## 📊 Features Engineering

### Self Features (Hành vi cá nhân)

1. **Geometry & Shape**

   - Body length, head width, elongation
   - Body curvature, tail curvature
   - Body compactness

2. **Dynamics & Motion**

   - Speed (body_center, nose, tail)
   - Acceleration
   - Head rotation speed
   - Vertical velocity
   - Direction change

3. **Posture**

   - Ear-to-nose distance
   - Body angle

4. **Rolling Statistics**
   - Mean, std, max của các features trên với windows [5, 15, 30, 60, 90, 120]
   - Immobility indicators

### Pair Features (Hành vi tương tác)

1. **Distance Features**

   - Khoảng cách giữa tất cả cặp body parts

2. **Angle Features**

   - Facing angles (agent -> target, target -> agent)
   - Relative position angles

3. **Speed Features**

   - Speed của ears, tail_base
   - Approach/escape speed

4. **Interaction Features**
   - Proximity duration
   - Elongation và body angle của cả 2 chuột

## 🤖 Model Architecture

- **Algorithm**: XGBoost (Gradient Boosting)
- **Training Strategy**: 3-Fold Stratified Group K-Fold Cross Validation
- **Hyperparameters**: Adaptive dựa trên class imbalance
  - **Rare behaviors** (<0.1% positive samples): Conservative params, shallow trees
  - **Common behaviors**: Deeper trees, more rounds
- **Threshold Tuning**: Grid search để maximize F1 score cho từng behavior

## 📈 Kết quả

### Overall Performance

- **Overall F1 Score**: 0.5052
- **Single Behaviors Avg F1**: 0.2544
- **Pair Behaviors Avg F1**: 0.3974

### Per-Behavior Performance
------------------------------------------------------------
Action               Mode       Count      Avg F1    
------------------------------------------------------------
allogroom            pair       17         0.1756    
approach             pair       258        0.3867    
attack               pair       389        0.5623    
attemptmount         pair       42         0.0720    
avoid                pair       136        0.1504    
biteobject           single     16         0.0196    
chase                pair       117        0.1614    
chaseattack          pair       22         0.1789    
climb                single     30         0.2666    
defend               pair       64         0.3966    
dig                  single     60         0.1498    
disengage            pair       20         0.4422    
dominance            pair       6          0.6304    
dominancegroom       pair       14         0.1594    
dominancemount       pair       63         0.3964    
ejaculate            pair       3          0.4706    
escape               pair       125        0.3186    
exploreobject        single     17         0.0370    
flinch               pair       22         0.0864    
follow               pair       53         0.4665    
freeze               single     9          0.3260    
genitalgroom         single     17         0.5062    
huddle               single     11         0.4678    
intromit             pair       81         0.7206    
mount                pair       247        0.6123    
rear                 single     137        0.2406    
reciprocalsniff      pair       42         0.6827    
rest                 single     21         0.1415    
run                  single     19         0.0000    
selfgroom            single     108        0.1751    
shepherd             pair       16         0.4169    
sniff                pair       621        0.6358    
sniffbody            pair       109        0.5094    
sniffface            pair       119        0.5476    
sniffgenital         pair       462        0.4838    
submit               pair       23         0.2642    
tussle               pair       6          0.3159    

## 🔧 Troubleshooting

### GPU Training Failed

Nếu GPU training thất bại, hệ thống sẽ tự động fallback sang CPU. Để kiểm tra CUDA:

```python
import xgboost as xgb
print(xgb.train.__doc__)  # Check xgboost version
```

### Memory Issues

Nếu bị out of memory:

- Dùng `--mode sequential` khi tạo features
- Giảm `ROLLING_WINDOWS` trong `config.py`
- Giảm `XGB_MAX_BIN` xuống 32 hoặc 16

### Missing Features

Nếu một số video không có features:

```bash
python scripts/01_prepare_features.py --check_only
```

Sau đó chạy lại bước 1 cho các video bị thiếu.

## 📚 Tài liệu tham khảo

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [MABe Challenge](https://www.aicrowd.com/challenges/mabe-2022-track-1)

## 👨‍💻 Liên hệ

Nếu có thắc mắc, vui lòng liên hệ:

- Email: [minh2m5@gmail.com]
- GitHub: [https://github.com/nNm205]

## 📄 License

MIT License - Dự án học tập, mã nguồn mở cho cộng đồng.

---

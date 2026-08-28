# -*- coding: utf-8 -*-
"""V2 全局配置：路径、列定义、阈值、特征筛选、时序场景参数"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(ROOT, "data")
WORKSPACE_DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
LEGACY_DATA_DIR = r"D:/download/08-智能风控与量化建模赛道-江苏银行-信贷风控模型智能监测与自主迭代"
# 优先使用项目自带 data/ 目录，确保打包后开箱即用
DATA_DIR = os.environ.get(
    "RISK_AGENT_DATA_DIR",
    PROJECT_DATA_DIR if os.path.exists(os.path.join(PROJECT_DATA_DIR, "train_data.csv"))
    else (WORKSPACE_DATA_DIR if os.path.exists(os.path.join(WORKSPACE_DATA_DIR, "train_data.csv"))
          else LEGACY_DATA_DIR),
)
TRAIN_CSV = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_data.csv")

SCENARIO_DIR = os.path.join(ROOT, "data", "scenarios")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "output")
EVENT_LOG = os.path.join(ROOT, "output", "events.jsonl")
DASHBOARD_DIR = os.path.join(ROOT, "dashboard")

for d in (SCENARIO_DIR, MODEL_DIR, REPORT_DIR, DASHBOARD_DIR):
    os.makedirs(d, exist_ok=True)

ID_COL, TIME_COL, LABEL_COL = "id_card", "apply_time", "is_bad"
EXCLUDE = [ID_COL, TIME_COL, LABEL_COL]

# ---------- V2 特征筛选（确定性剔除） ----------
# 强特征剔除：单特征 AUC>0.80 的规则型信号特征
DROP_STRONG_FEATURES = ["login_fail_count", "max_overdue_days"]
# 共线性剔除：|corr|>0.80 保留 gain 高的
DROP_COLLINEAR_FEATURES = ["consumption_level"]
# 最终保留特征数 = 34 - 2 - 1 = 31
DROP_FEATURES = DROP_STRONG_FEATURES + DROP_COLLINEAR_FEATURES

# 类别/低基数特征
CATEGORICAL_COLS = ["blacklist_hit", "gps_anomaly", "device_type", "emulator_flag",
                    "education_level", "marital_status", "gender", "city_tier",
                    "repayment_period"]

# m1 §2.2 有效性校验：字段合法区间/枚举
VALID_RANGE = {
    "credit_utilization": (0, 1), "social_score": (300, 900),
    "telecom_score": (0, 100), "ecomm_risk_score": (0, 100),
    "judicial_risk_score": (0, 100), "night_activity_ratio": (0, 1),
    "device_risk_score": (0, 1), "age": (18, 65),
    "income_level": (2000, None), "debt_income_ratio": (0, 2),
    "loan_amount_request": (1000, None),
}
VALID_ENUM = {
    "blacklist_hit": {0, 1}, "gps_anomaly": {0, 1}, "device_type": {0, 1, 2},
    "emulator_flag": {0, 1}, "education_level": {1, 2, 3, 4, 5},
    "marital_status": {0, 1, 2}, "gender": {0, 1},
    "city_tier": {1, 2, 3, 4}, "repayment_period": {3, 6, 12, 24, 36},
}

# m2 阈值
TH = {
    "R1_ks_drop":      {"low": 0.05, "medium": 0.15},
    "R2_auc_drop":     {"low": 0.03, "medium": 0.10},
    "R3_max_feat_psi": {"low": 0.10, "medium": 0.25},
    "R4_group_psi":    {"low": 0.10, "medium": 0.30},
    "R5_score_psi":    {"low": 0.10, "medium": 0.25},
    "R6_bad_rate_pp":  {"low": 1.0, "medium": 2.0},
    "KS_ok": 0.35, "AUC_ok": 0.75,
    "KS_leak": 0.50, "AUC_leak": 0.90,
}

BASELINE_BAD_RATE = 0.04

# ---------- 模型参数 ----------
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=4,
                  learning_rate=0.03, subsample=0.7, colsample_bytree=0.6,
                  min_child_weight=100, reg_lambda=4.0, n_estimators=400, random_state=42)
LGB_PARAMS = dict(objective="binary", max_depth=4, num_leaves=15, learning_rate=0.03,
                  subsample=0.7, colsample_bytree=0.6, min_child_samples=100,
                  reg_lambda=4.0, reg_alpha=0.1, n_estimators=400, random_state=42, verbose=-1)
LR_PARAMS = dict(max_iter=1000, C=1.0)

TRAIN_FRAC = 0.7
LITE_LR = 0.03

# ---------- 分群维度 ----------
SEGMENT_COLS = {"age": [17, 25, 35, 45, 66],
                "loan_amount_request": None,
                "city_tier": None,
                "repayment_period": None}

FLIP_MARGIN_Q = 0.05
SCORE_THRESHOLD = 0.5

# ---------- V2 时序场景参数 ----------
# V4：验证周期统一为三个月（2026-01/02/03），每月 2 万条，标签延迟 30 天
TEMPORAL_MONTHS = ["2026-01", "2026-02", "2026-03"]
TEMPORAL_SAMPLE_SIZE = 20000       # 每月 2 万条
LABEL_DELAY_DAYS = 30              # 标签延迟 30 天
DRIFT_RATE_PER_MONTH = 0.08        # 每月漂移 8% 递增（月1=8%，月2=16%，月3=24%）

# 统一未来评测：2026-02~03（共 4 万条）
UNIFIED_EVAL_MONTHS = ["2026-02", "2026-03"]
# 适应窗口：2026-01（标签延迟 30 天，可观测不训练）
ADAPTATION_MONTH = "2026-01"

# Champion 替换阈值
DELONG_P_THRESHOLD = 0.05          # DeLong p<0.05 为显著
KS_SIGNIFICANT_DIFF = 0.02         # KS 差>+0.02 为显著优

# ---------- V3 场景规格：1次OOT + 4场景(A/B/C/D) ----------
# A=无漂移 / B=轻量特征漂移 / C=中度标签漂移 / D=重度联合漂移
# 每场景按月推进至4月(2026-01~04)，每月2万条，标签延迟30天
# 漂移注入每月8%递增（累计 8%/16%/24%）
# 删去 data_repair 策略，仅保留 light/standard/major + none
SCENARIO_SPEC = {
    "oot_2026_01": {
        "desc": "真实OOT验证",
        "feature_drift": {}, "label_flip": 0.0, "missing_inject": {},
        "expect_alarm": "MEDIUM", "expect_strategy": "light",
        "is_oot": True,
    },
    "parallel_A": {
        "desc": "无漂移场景",
        "feature_drift": {}, "label_flip": 0.0, "missing_inject": {},
        "expect_alarm": "LOW", "expect_strategy": "none",
    },
    "parallel_B": {
        "desc": "轻量特征漂移",
        "feature_drift": {
            "age": {"mode": "mean_shift", "base_intensity": -3.0},
            "city_tier": {"mode": "prob_shift", "base_intensity": 0.15},
            "income_level": {"mode": "mean_shift", "base_intensity": -2000.0},
        },
        "label_flip": 0.0, "missing_inject": {},
        "expect_alarm": "MEDIUM", "expect_strategy": "light",
        "round_drift_factors": [0.0, 0.05, 0.4],
    },
    "parallel_C": {
        "desc": "中度标签漂移",
        "feature_drift": {}, "label_flip": 0.003,
        "missing_inject": {},
        "expect_alarm": "MEDIUM", "expect_strategy": "standard",
    },
    "parallel_D": {
        "desc": "重度联合漂移",
        "feature_drift": {
            "age": {"mode": "mean_shift", "base_intensity": -3.0},
            "city_tier": {"mode": "prob_shift", "base_intensity": 0.20},
            "income_level": {"mode": "mean_shift", "base_intensity": -2000.0},
            "credit_query_times": {"mode": "mean_shift", "base_intensity": 0.5},
            "social_score": {"mode": "mean_shift", "base_intensity": -15.0},
        },
        "label_flip": 0.008,
        "missing_inject": {},
        "expect_alarm": "HIGH", "expect_strategy": "major",
        "round_drift_factors": [0.05, 0.15, 1.0],
    },
}

# V3 场景列表（含OOT + A/B/C/D）
ALL_SCENARIOS = ["oot_2026_01", "parallel_A", "parallel_B", "parallel_C", "parallel_D"]
# 时序场景（不含OOT，OOT用官方test单月）
TEMPORAL_SCENARIOS = ["parallel_A", "parallel_B", "parallel_C", "parallel_D"]

# -*- coding: utf-8 -*-
"""模型训练/保存/加载/打分统一封装。一个"模型制品" = {algo, model, features}"""
import os, json, pickle, time
import numpy as np
import pandas as pd
from . import config as C
from .metrics import ks_score, auc_score


def load_dataframe(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in C.EXCLUDE]


def time_split(df, frac=0.7, time_col=None):
    """时间外验证切分（OOT 语义）：按 apply_time 升序，前 frac 训练、其余验证。
    用"过去"预测"未来"，替代随机抽样，更贴近真实上线的时间外泛化口径。"""
    tc = time_col or C.TIME_COL
    d = df.sort_values(tc).reset_index(drop=True)
    n = int(len(d) * frac)
    return d.iloc[:n], d.iloc[n:]


def train_model(algo, X, y, params=None, sample_weight=None, init_model=None):
    """统一训练入口。
    sample_weight: 每样本权重（用于 standard 的标签先验重加权）。
    init_model:   已有 booster，用于 light 的增量微调（在 v1 之上追加树，warm_start）。
    """
    params = params or {}
    if algo == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(**{**C.XGB_PARAMS, **params})
    elif algo == "lgb":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(**{**C.LGB_PARAMS, **params})
    elif algo == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(**{**C.LR_PARAMS, **params}))
    else:
        raise ValueError(algo)
    t0 = time.time()
    fit_kwargs = {}
    if sample_weight is not None and algo != "lr":
        fit_kwargs["sample_weight"] = sample_weight
    if init_model is not None and algo == "lgb":
        fit_kwargs["init_model"] = init_model       # lgb 增量：在既有 booster 上继续
    elif init_model is not None and algo == "xgb":
        fit_kwargs["xgb_model"] = init_model        # xgb 增量
    m.fit(X, y, **fit_kwargs)
    return m, time.time() - t0


def predict(artifact, X):
    return artifact["model"].predict_proba(X)[:, 1]


def save_artifact(path, algo, model, features, meta=None):
    with open(path, "wb") as f:
        pickle.dump({"algo": algo, "model": model, "features": features, "meta": meta or {}}, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(artifact, df):
    X = df[artifact["features"]]
    s = predict(artifact, X)
    return {"KS": ks_score(df[C.LABEL_COL], s), "AUC": auc_score(df[C.LABEL_COL], s),
            "score": s}


def select_best(candidates):
    """m4 §6 选优（门控拆分版）。

    两个指标独立展示、不再混用单一 train_ks − oot_ks：
      - 过拟合 gap（同源内验）：train_ks − va_ks。衡量模型是否"背"训练数据。
        训练与验证同分布（均来自 train 前/后 70%），差值即真过拟合。门控降低至 ≤0.03。
      - 漂移适应度 va→oot 衰减：va_ks − oot_ks。衡量模型跨分布（验证集→OOT/场景）
        的泛化落差。衰减主要来自分布漂移而非过拟合，仅展示不进过拟合门控。

    若候选缺少 va_ks（未走同源内验），退化用 train_ks − oot_ks 作为过拟合 gap 兜底。
    candidates: [{name, artifact, train_ks, va_ks?, oot_ks, oot_auc}]
    """
    GATE_OVERFIT = 0.03   # 降低后的真过拟合门控（同源 train→va）
    pool = []
    for c in candidates:
        va_ks = c.get("va_ks", c["oot_ks"])          # 无内验时以 oot 兜底
        c["overfit_gap"] = c["train_ks"] - va_ks      # 真过拟合（同源）
        c["drift_adapt_gap"] = va_ks - c["oot_ks"]    # va→OOT 漂移适应度（仅展示）
        if c["overfit_gap"] <= GATE_OVERFIT:
            pool.append(c)
    if not pool:                      # 全部过拟合门控失败：选 OOT KS 最高者兜底并标记（业务可用性优先）
        best = max(candidates, key=lambda c: c["oot_ks"])
        best["gate_pass"] = False
        best["score"] = 0.0
        return best
    ks_max = max(c["oot_ks"] for c in pool) or 1
    auc_max = max(c["oot_auc"] for c in pool) or 1
    for c in pool:
        c["score"] = 0.5 * c["oot_ks"] / ks_max + 0.5 * c["oot_auc"] / auc_max
        c["gate_pass"] = True
    pool.sort(key=lambda c: (c["score"], c["oot_ks"]), reverse=True)
    return pool[0]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

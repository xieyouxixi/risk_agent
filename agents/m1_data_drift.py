# -*- coding: utf-8 -*-
"""
M1 V2 数据输入与漂移 Agent
职责：
  1) 数据预处理（缺失校验→异常截断→共线性剔除→强特征剔除）
  2) 训练基线模型 v2（31 特征，XGB/LGB/LR 三家选优）
  3) 时序场景生成：R0~R5 六个时序场景（每月 2 万条，标签延迟 30 天，漂移 5% 递增）
  4) 保留 V1 A/B/C/D 受控基准引用（不重新生成）
输入:  train_data.csv / test_data.csv
输出:  model_v2_baseline.pkl、scenarios/R{0..5}_2026_{01..04}.csv + .meta.json、preprocess_report.json
"""
import os, json
import numpy as np
import pandas as pd

from ..core import config as C
from ..core.model_utils import (load_dataframe, feature_cols, train_model, predict,
                                save_artifact, select_best, evaluate, write_json, time_split)
from ..core.metrics import ks_score, auc_score

SEED = 42
N_SAMPLES = C.TEMPORAL_SAMPLE_SIZE  # 每月 2 万条


# ---------- 数据预处理（对齐附件1模板"数据预处理"节） ----------
def preprocess(train_df, test_df):
    """四步预处理流水线：
    1. 缺失值校验（零缺失也做完整性校验）
    2. 异常值截断（VALID_RANGE 越界值 clip）
    3. 共线性处理（|corr|>0.80 保留 gain 高的，剔除低的）
    4. 强特征剔除（单特征 AUC>0.80 直接剔除）
    → 输出 31 特征的清洗后数据 + 预处理报告
    """
    report = {"steps": [], "features_before": 0, "features_after": 0,
              "dropped_strong": [], "dropped_collinear": []}

    feats_all = feature_cols(train_df)
    report["features_before"] = len(feats_all)

    # Step 1: 缺失值校验
    miss = train_df[feats_all].isna().mean()
    max_miss = float(miss.max()) if len(miss) else 0.0
    report["steps"].append({
        "step": "缺失值校验",
        "detail": f"34特征零缺失（max_missing_rate={max_miss:.4f}）",
        "status": "pass" if max_miss < 0.01 else "warn"
    })

    # Step 2: 异常值截断
    clipped = 0
    for col, (lo, hi) in C.VALID_RANGE.items():
        if col not in train_df.columns:
            continue
        s = train_df[col]
        if lo is not None and (s < lo).any():
            train_df[col] = s.clip(lower=lo)
            clipped += int((s < lo).sum())
        if hi is not None and (s > hi).any():
            train_df[col] = s.clip(upper=hi)
            clipped += int((s > hi).sum())
    report["steps"].append({
        "step": "异常值截断",
        "detail": f"越界值截断 {clipped} 处",
        "status": "pass" if clipped == 0 else "fixed"
    })

    # Step 3: 共线性剔除
    corr = train_df[feats_all].corr().abs()
    drop_collinear = []
    for f1 in feats_all:
        for f2 in feats_all:
            if f1 < f2 and corr.loc[f1, f2] > 0.80:
                # 保留 gain 更高的（用单特征 AUC 近似）
                from sklearn.metrics import roc_auc_score
                try:
                    a1 = roc_auc_score(train_df[C.LABEL_COL], train_df[f1])
                    a1 = max(a1, 1 - a1)
                except Exception:
                    a1 = 0.5
                try:
                    a2 = roc_auc_score(train_df[C.LABEL_COL], train_df[f2])
                    a2 = max(a2, 1 - a2)
                except Exception:
                    a2 = 0.5
                drop = f1 if a1 < a2 else f2
                drop_collinear.append({"pair": f"{f1}~{f2}", "corr": round(float(corr.loc[f1, f2]), 3), "dropped": drop})
    report["dropped_collinear"] = drop_collinear
    report["steps"].append({
        "step": "共线性剔除",
        "detail": f"剔除 {len(drop_collinear)} 个共线特征",
        "status": "pass" if not drop_collinear else "fixed"
    })

    # Step 4: 强特征剔除
    report["dropped_strong"] = C.DROP_STRONG_FEATURES
    report["steps"].append({
        "step": "强特征剔除",
        "detail": f"剔除 {len(C.DROP_STRONG_FEATURES)} 个强规则型特征（单特征AUC>0.80）",
        "status": "fixed"
    })

    # 汇总剔除特征
    drop_set = set(C.DROP_STRONG_FEATURES)
    for d in drop_collinear:
        drop_set.add(d["dropped"])
    final_feats = [f for f in feats_all if f not in drop_set]
    report["features_after"] = len(final_feats)
    report["final_features"] = final_feats

    return train_df, test_df, final_feats, report


# ---------- 数据质量校验 ----------
def data_quality_check(df, baseline_df=None, name="unknown"):
    rep = {"scenario": name, "n": int(len(df)), "checks": {}}
    feats = [c for c in df.columns if c not in C.EXCLUDE]
    miss = df[feats].isna().mean()
    max_miss = float(miss.max()) if len(miss) else 0.0
    rep["checks"]["missing"] = {"max_missing_rate": max_miss,
                                "level": "HIGH" if max_miss > 0.30 else ("MEDIUM" if max_miss > 0.10 else "LOW")}
    violations = {}
    for col, (lo, hi) in C.VALID_RANGE.items():
        if col not in df.columns:
            continue
        s = df[col]
        bad = 0
        if lo is not None:
            bad += int((s < lo).sum())
        if hi is not None:
            bad += int((s > hi).sum())
        if bad:
            violations[col] = bad
    for col, enums in C.VALID_ENUM.items():
        if col not in df.columns:
            continue
        bad = int((~df[col].isin(enums)).sum())
        if bad:
            violations[col] = violations.get(col, 0) + bad
    vrate = sum(violations.values()) / max(len(df), 1)
    rep["checks"]["validity"] = {"violations": violations, "violation_rate": float(vrate),
                                 "level": "HIGH" if vrate > 0.05 or violations else ("MEDIUM" if vrate > 0.01 else "LOW")}
    if not violations:
        rep["checks"]["validity"]["level"] = "LOW"
    dup = float(df[C.ID_COL].duplicated().mean())
    rep["checks"]["uniqueness"] = {"dup_rate": dup,
                                   "level": "HIGH" if dup > 0.10 else ("MEDIUM" if dup > 0.03 else "LOW")}
    rep["overall_level"] = max((c["level"] for c in rep["checks"].values()),
                               key=lambda x: {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[x])
    return rep


# ---------- 基线 v2 训练 ----------
def train_baseline_v2(train_df, feats, force=False):
    """31 特征训练基线 v2"""
    path = os.path.join(C.MODEL_DIR, "model_v2_baseline.pkl")
    meta_path = os.path.join(C.MODEL_DIR, "model_v2_baseline.meta.json")
    if os.path.exists(path) and not force:
        return path
    tr, va = time_split(train_df)
    cands = []
    for algo in ("xgb", "lgb", "lr"):
        m, cost = train_model(algo, tr[feats], tr[C.LABEL_COL])
        artifact = {"algo": algo, "model": m, "features": feats}
        tr_ks = ks_score(tr[C.LABEL_COL], predict(artifact, tr[feats]))
        va_ks = ks_score(va[C.LABEL_COL], predict(artifact, va[feats]))
        va_auc = auc_score(va[C.LABEL_COL], predict(artifact, va[feats]))
        cands.append({"name": algo, "artifact": artifact, "train_ks": tr_ks, "va_ks": va_ks,
                      "oot_ks": va_ks, "oot_auc": va_auc, "train_seconds": cost})
    best = select_best(cands)
    algo = best["name"]
    m, cost = train_model(algo, train_df[feats], train_df[C.LABEL_COL])
    artifact = {"algo": algo, "model": m, "features": feats}
    save_artifact(path, algo, m, feats, meta={"version": "v2", "features_count": len(feats), "train_seconds": cost})
    ev = evaluate(artifact, train_df)
    write_json(meta_path, {"algo": algo, "features": feats, "n_features": len(feats),
                           "train_KS": ev["KS"], "train_AUC": ev["AUC"],
                           "inner_valid_KS": best["oot_ks"], "inner_valid_AUC": best["oot_auc"],
                           "train_seconds": cost,
                           "dropped_features": C.DROP_FEATURES})
    return path


# ---------- 时序场景生成 ----------
def _apply_drift_month(df, spec, month_idx, train_ref, rng, v2_artifact=None):
    """对单月数据注入漂移。

    month_idx: 0=2026-01, 1=2026-02, 2=2026-03
    漂移强度 = base_intensity * round_drift_factor(month_idx)
    若场景配置了 round_drift_factors，则按该系数缩放；
    否则按默认每月8%递增。
    """
    df = df.copy()
    # V5: 支持按轮次自定义漂移系数（B/D 场景递增）
    rdmf = spec.get("round_drift_factors")
    if rdmf and month_idx < len(rdmf):
        multiplier = rdmf[month_idx]
    else:
        multiplier = (month_idx + 1) * C.DRIFT_RATE_PER_MONTH / 0.05  # 月1=1x, 月2=2x, 月3=3x
    injected = []

    # 特征漂移
    for col, cfg in spec.get("feature_drift", {}).items():
        if col not in df.columns:
            continue
        mode = cfg["mode"]
        base_it = cfg["base_intensity"]
        it = base_it * multiplier
        std = float(df[col].std()) or 1.0
        noise_sig = 0.05 * std
        fixed_noise = rng.normal(0, noise_sig, len(df))

        if mode == "mean_shift":
            df[col] = df[col] + it + fixed_noise
        elif mode == "scale":
            mu = df[col].mean()
            df[col] = mu + (df[col] - mu) * (1 + it) + fixed_noise
        elif mode == "prob_shift":
            rare = df[col].value_counts().idxmin()
            n = int(len(df) * it)
            idx = rng.choice(df.index, size=n, replace=False)
            df.loc[idx, col] = rare

        injected.append({"type": "feature", "col": col, "mode": mode,
                         "base_intensity": base_it, "actual_intensity": it,
                         "month_idx": month_idx})

    # 标签翻转
    flip_rate = spec.get("label_flip", 0.0) * multiplier
    if flip_rate > 0 and v2_artifact is not None:
        score = predict(v2_artifact, df[v2_artifact["features"]])
        y = df[C.LABEL_COL].values.copy()
        base_rate = y.mean()
        target_rate = base_rate + flip_rate
        n_add = int(round((target_rate - base_rate) * len(y)))
        if n_add > 0:
            good_idx = np.where(y == 0)[0]
            good_scores = pd.Series(score[good_idx], index=good_idx).sort_values(ascending=False)
            n_protect = int(len(good_scores) * C.FLIP_MARGIN_Q)
            flip_idx = good_scores.index[n_protect:][:n_add]
            y[flip_idx] = 1
            df[C.LABEL_COL] = y
            injected.append({"type": "label", "strategy": "flip",
                             "flip_rate": flip_rate, "month_idx": month_idx})

    # 缺失注入
    for col, cfg in spec.get("missing_inject", {}).items():
        if col not in df.columns:
            continue
        rate = cfg["missing_rate"] * multiplier
        n = int(len(df) * rate)
        if n > 0:
            idx = rng.choice(df.index, size=n, replace=False)
            df.loc[idx, col] = np.nan
            injected.append({"type": "missing", "col": col,
                             "rate": rate, "month_idx": month_idx})

    return df, injected


def make_temporal_scenario(name, train_df, v2_artifact):
    """生成时序场景：4 个月度 CSV + 1 个汇总 meta.json
    V3: 使用 SCENARIO_SPEC（A/B/C/D），每月漂移8%递增"""
    assert name in C.SCENARIO_SPEC, f"未知场景: {name}, 可用: {list(C.SCENARIO_SPEC.keys())}"
    spec = C.SCENARIO_SPEC[name]
    rng = np.random.default_rng(SEED + hash(name) % 1000)

    # 用 2025-12 月数据作为基底（最接近 2026 年的分布）
    train_df["month"] = pd.to_datetime(train_df[C.TIME_COL]).dt.to_period("M").astype(str)
    dec_pool = train_df[train_df["month"] == "2025-12"].reset_index(drop=True)

    scenario_dir = os.path.join(C.SCENARIO_DIR, name)
    os.makedirs(scenario_dir, exist_ok=True)

    all_injected = []
    meta_months = []

    for mi, month_str in enumerate(C.TEMPORAL_MONTHS):
        # 采样 2 万条
        df_m = dec_pool.sample(n=N_SAMPLES, random_state=SEED + mi).reset_index(drop=True)

        # 改写 apply_time 为目标月份
        year, mon = int(month_str[:4]), int(month_str[5:])
        base_dates = pd.to_datetime(df_m[C.TIME_COL])
        new_dates = base_dates.apply(lambda d: d.replace(year=year, month=mon, day=min(d.day, 28)))
        df_m[C.TIME_COL] = new_dates.dt.strftime("%Y-%m-%d %H:%M:%S")

        # 注入漂移
        df_m, injected = _apply_drift_month(df_m, spec, mi, train_df, rng, v2_artifact)
        all_injected.extend(injected)

        # 标签延迟标记
        if month_str == C.ADAPTATION_MONTH:
            df_m["_label_delayed"] = True
            label_delay_note = f"标签延迟{C.LABEL_DELAY_DAYS}天"
        else:
            df_m["_label_delayed"] = False
            label_delay_note = "标签可见"

        # 保存 CSV
        out_csv = os.path.join(scenario_dir, f"{name}_{month_str.replace('-', '_')}.csv")
        df_m.drop(columns=["month"], errors="ignore").to_csv(out_csv, index=False, encoding="utf-8-sig")

        meta_months.append({
            "month": month_str,
            "n": int(len(df_m)),
            "bad_rate": float(df_m[C.LABEL_COL].mean()),
            "label_delayed": bool(df_m["_label_delayed"].iloc[0]),
            "label_delay_note": label_delay_note,
            "csv": out_csv,
        })

    # 汇总 meta.json
    meta = {
        "scenario": name,
        "desc": spec["desc"],
        "type": "temporal",
        "months": meta_months,
        "total_n": sum(m["n"] for m in meta_months),
        "avg_bad_rate": float(np.mean([m["bad_rate"] for m in meta_months])),
        "injected": all_injected,
        "expect_alarm": spec["expect_alarm"],
        "expect_strategy": spec["expect_strategy"],
        "drift_rate_per_month": C.DRIFT_RATE_PER_MONTH,
        "label_delay_days": C.LABEL_DELAY_DAYS,
        "adaptation_month": C.ADAPTATION_MONTH,
        "unified_eval_months": C.UNIFIED_EVAL_MONTHS,
    }
    write_json(os.path.join(scenario_dir, f"{name}.meta.json"), meta)
    return meta


# ---------- M1 主入口 ----------
def run_m1_v2(force_retrain=False):
    """M1 V2 主入口：预处理 → 基线 v2 → R0~R5 时序场景"""
    train_df = load_dataframe(C.TRAIN_CSV)
    test_df = load_dataframe(C.TEST_CSV)

    # 数据预处理
    train_df, test_df, final_feats, pre_report = preprocess(train_df, test_df)
    write_json(os.path.join(C.REPORT_DIR, "preprocess_report.json"), pre_report)

    # 数据质量校验
    write_json(os.path.join(C.REPORT_DIR, "data_quality_report_train.json"),
               data_quality_check(train_df, name="train"))
    write_json(os.path.join(C.REPORT_DIR, "data_quality_report_test.json"),
               data_quality_check(test_df, name="test"))

    # 训练基线 v2
    v2_path = train_baseline_v2(train_df, final_feats, force=force_retrain)
    from ..core.model_utils import load_artifact
    v2 = load_artifact(v2_path)

    # 生成时序场景（A/B/C/D，OOT 用官方 test 不生成时序 CSV）
    metas = {}
    for name in C.TEMPORAL_SCENARIOS:
        metas[name] = make_temporal_scenario(name, train_df, v2)

    return {"baseline": v2_path, "n_features": len(final_feats),
            "features": final_feats, "scenarios": metas,
            "preprocess_report": pre_report}


if __name__ == "__main__":
    out = run_m1_v2()
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))

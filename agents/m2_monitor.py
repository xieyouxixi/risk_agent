# -*- coding: utf-8 -*-
"""
M2 V2 监测 Agent（答题人·第一道防线）
职责：双阶段监测——标签未成熟期（无标签）+ 标签成熟后（有标签）
输入:  v2 基线模型 + 基线数据(train) + 当月数据(scenario csv)
输出:  monitor_report_{scenario}.json（指标值 + 每条规则等级 + 综合告警等级）
"""
import os
import numpy as np
import pandas as pd

from ..core import config as C
from ..core.model_utils import load_dataframe, load_artifact, predict, write_json
from ..core import model_registry
from ..core.metrics import ks_score, auc_score, psi, psi_categorical, bad_rate


def _level(value, th):
    if value > th["medium"]:
        return "HIGH"
    if value > th["low"]:
        return "MEDIUM"
    return "LOW"


class MonitorAgentV2:
    def __init__(self, baseline_model_path=None, baseline_df=None):
        # 按月循环：不传 baseline_model_path 时从注册表取当前 champion
        if baseline_model_path is None:
            baseline_model_path = model_registry.current_champion_path()
        self.champion_path = baseline_model_path
        self.v2 = load_artifact(baseline_model_path)
        self.base_df = baseline_df if baseline_df is not None else load_dataframe(C.TRAIN_CSV)
        s_base = predict(self.v2, self.base_df[self.v2["features"]])
        self.base = {
            "KS": ks_score(self.base_df[C.LABEL_COL], s_base),
            "AUC": auc_score(self.base_df[C.LABEL_COL], s_base),
            "bad_rate": bad_rate(self.base_df[C.LABEL_COL]),
            "score": s_base,
        }

    def _feature_psi_table(self, cur_df):
        rows = []
        for col in self.v2["features"]:
            if col not in cur_df.columns:
                continue
            b, a = self.base_df[col], cur_df[col]
            if col in C.CATEGORICAL_COLS or b.nunique() <= 10:
                v = psi_categorical(b, a)
            else:
                v = psi(b, a)
            rows.append({"feature": col, "psi": v})
        t = pd.DataFrame(rows).sort_values("psi", ascending=False)
        return t

    def monitor_no_label(self, cur_df, scenario="unknown"):
        """标签未成熟期：无标签监测（只观测特征分布、评分分布、客群结构、数据质量）"""
        feats = self.v2["features"]
        s_cur = predict(self.v2, cur_df[feats])
        score_psi = psi(self.base["score"], s_cur)
        psi_tab = self._feature_psi_table(cur_df)
        max_feat_psi = float(psi_tab["psi"].max()) if len(psi_tab) else 0.0
        group_ratio = float((psi_tab["psi"] > 0.10).mean()) if len(psi_tab) else 0.0

        # 数据质量
        miss = cur_df[feats].isna().mean()
        max_miss = float(miss.max()) if len(miss) else 0.0
        miss_count = int((miss > 0.01).sum())

        rules = {
            "R3_最大特征PSI": {"value": max_feat_psi, "level": _level(max_feat_psi, C.TH["R3_max_feat_psi"])},
            "R4_群体漂移占比": {"value": group_ratio, "level": _level(group_ratio, C.TH["R4_group_psi"])},
            "R5_模型分PSI": {"value": score_psi, "level": _level(score_psi, C.TH["R5_score_psi"])},
            "DQ_缺失率": {"value": max_miss,
                         "level": "HIGH" if max_miss > 0.30 else ("MEDIUM" if max_miss > 0.10 else "LOW")},
            "DQ_缺失特征数": {"value": miss_count,
                           "level": "HIGH" if miss_count > 5 else ("MEDIUM" if miss_count > 2 else "LOW")},
        }
        order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        overall = max((r["level"] for r in rules.values()), key=lambda x: order[x])
        fired = [k for k, r in rules.items() if r["level"] in ("MEDIUM", "HIGH")]

        rep = {
            "scenario": scenario,
            "phase": "no_label",
            "n": int(len(cur_df)),
            "metrics": {"score_psi": score_psi, "max_feat_psi": max_feat_psi,
                        "group_psi_ratio": group_ratio,
                        "max_missing_rate": max_miss, "missing_feat_count": miss_count},
            "baseline": {"KS": self.base["KS"], "AUC": self.base["AUC"],
                         "bad_rate": self.base["bad_rate"]},
            "rules": rules, "overall_level": overall, "fired_rules": fired,
            "feature_psi_top10": psi_tab.head(10).to_dict("records"),
            "note": f"标签延迟{C.LABEL_DELAY_DAYS}天，当前为无标签监测阶段",
        }
        write_json(os.path.join(C.REPORT_DIR, f"monitor_report_{scenario}.json"), rep)
        return rep

    def monitor_with_label(self, cur_df, scenario="unknown"):
        """标签成熟后：有标签监测（完整 R1~R7）"""
        feats = self.v2["features"]
        s_cur = predict(self.v2, cur_df[feats])
        y = cur_df[C.LABEL_COL].values

        ks_cur = ks_score(y, s_cur)
        auc_cur = auc_score(y, s_cur)
        br_cur = bad_rate(y)
        score_psi = psi(self.base["score"], s_cur)
        psi_tab = self._feature_psi_table(cur_df)
        max_feat_psi = float(psi_tab["psi"].max()) if len(psi_tab) else 0.0
        group_ratio = float((psi_tab["psi"] > 0.10).mean()) if len(psi_tab) else 0.0

        ks_drop = (self.base["KS"] - ks_cur) / self.base["KS"]
        auc_drop = (self.base["AUC"] - auc_cur) / self.base["AUC"]
        br_delta_pp = (br_cur - self.base["bad_rate"]) * 100

        # 数据质量
        miss = cur_df[feats].isna().mean()
        max_miss = float(miss.max()) if len(miss) else 0.0
        miss_count = int((miss > 0.01).sum())

        rules = {
            "R1_KS下降率": {"value": ks_drop, "level": _level(ks_drop, C.TH["R1_ks_drop"])},
            "R2_AUC下降率": {"value": auc_drop, "level": _level(auc_drop, C.TH["R2_auc_drop"])},
            "R3_最大特征PSI": {"value": max_feat_psi, "level": _level(max_feat_psi, C.TH["R3_max_feat_psi"])},
            "R4_群体漂移占比": {"value": group_ratio, "level": _level(group_ratio, C.TH["R4_group_psi"])},
            "R5_模型分PSI": {"value": score_psi, "level": _level(score_psi, C.TH["R5_score_psi"])},
            "R6_坏账率变化pp": {"value": br_delta_pp, "level": _level(abs(br_delta_pp), C.TH["R6_bad_rate_pp"])},
            "R7_坏账率趋势": {"value": br_delta_pp,
                           "level": "HIGH" if br_delta_pp > 2 else ("MEDIUM" if 1 < br_delta_pp <= 2 else "LOW")},
            "DQ_缺失率": {"value": max_miss,
                         "level": "HIGH" if max_miss > 0.30 else ("MEDIUM" if max_miss > 0.10 else "LOW")},
            "DQ_缺失特征数": {"value": miss_count,
                           "level": "HIGH" if miss_count > 5 else ("MEDIUM" if miss_count > 2 else "LOW")},
        }
        order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        overall = max((r["level"] for r in rules.values()), key=lambda x: order[x])
        fired = [k for k, r in rules.items() if r["level"] in ("MEDIUM", "HIGH")]

        rep = {
            "scenario": scenario,
            "phase": "with_label",
            "n": int(len(cur_df)),
            "metrics": {"KS": ks_cur, "AUC": auc_cur, "bad_rate": br_cur,
                        "score_psi": score_psi, "max_feat_psi": max_feat_psi,
                        "group_psi_ratio": group_ratio,
                        "ks_drop_rate": ks_drop, "auc_drop_rate": auc_drop,
                        "bad_rate_delta_pp": br_delta_pp,
                        "max_missing_rate": max_miss, "missing_feat_count": miss_count},
            "baseline": {"KS": self.base["KS"], "AUC": self.base["AUC"],
                         "bad_rate": self.base["bad_rate"]},
            "rules": rules, "overall_level": overall, "fired_rules": fired,
            "feature_psi_top10": psi_tab.head(10).to_dict("records"),
        }
        write_json(os.path.join(C.REPORT_DIR, f"monitor_report_{scenario}.json"), rep)
        return rep

    def monitor_month(self, df, scenario, month, has_label=True, champion_path=None):
        """按月监测入口：写 monitor_report_{scenario}_{month}.json，并附 champion/月份上下文。"""
        rep = self.monitor_with_label(df, scenario) if has_label else self.monitor_no_label(df, scenario)
        rep["month"] = month
        rep["champion"] = os.path.basename(self.champion_path)
        # 覆盖写带月份的报告（不覆盖整场景报告）
        rep_file = f"monitor_report_{scenario}_{month.replace('-', '_')}.json"
        write_json(os.path.join(C.REPORT_DIR, rep_file), rep)
        return rep


def run_m2_v2(scenario, csv_path=None, has_label=True, month=None, champion_path=None):
    """V2 监测入口：month=None 走兼容旧路径（适应窗口），month='2026-01'/'2026-02'... 走按月循环"""
    if csv_path is None:
        if scenario == "oot_2026_01":
            csv_path = C.TEST_CSV
        else:
            m = month or C.ADAPTATION_MONTH
            csv_path = os.path.join(C.SCENARIO_DIR, scenario,
                                    f"{scenario}_{m.replace('-', '_')}.csv")
    df = load_dataframe(csv_path)
    agent = MonitorAgentV2(baseline_model_path=champion_path)
    if month is not None:
        return agent.monitor_month(df, scenario, month, has_label=has_label)
    if has_label:
        return agent.monitor_with_label(df, scenario=scenario)
    else:
        return agent.monitor_no_label(df, scenario=scenario)

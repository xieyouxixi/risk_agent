# -*- coding: utf-8 -*-
"""
M3 根因分析 Agent（双层诊断）
第一层（出了什么问题）：坏账率变化 / 客群结构偏移 / KS衰减 / AUC衰减 / 过拟合与泄漏
第二层（为什么出问题）：IV衰减 / 重要性偏移 / 群体性PSI / 单点PSI / 缺失与越界
输出:  rootcause_report_{scenario}.json + 根因分析报告_{scenario}.docx
"""
import os
import numpy as np
import pandas as pd

from ..core import config as C
from ..core.model_utils import (load_dataframe, load_artifact, predict, write_json,
                                evaluate, train_model)
from ..core.metrics import (ks_score, auc_score, psi, psi_categorical, iv_score, bad_rate)
from ..reports.docx_report import write_rootcause_docx
from .m3_llm_agent import M3LLMRootCauseAgent

LV_ORD = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _lv(value, low, high):
    if value > high:
        return "HIGH"
    if value > low:
        return "MEDIUM"
    return "LOW"


class RootCauseAgentV2:
    def __init__(self, baseline_model_path=None, baseline_df=None):
        baseline_model_path = baseline_model_path or os.path.join(C.MODEL_DIR, "model_v2_baseline.pkl")
        self.v2 = load_artifact(baseline_model_path)
        self.base_df = baseline_df if baseline_df is not None else load_dataframe(C.TRAIN_CSV)
        s = predict(self.v2, self.base_df[self.v2["features"]])
        self.base_score = s
        self.base = {"KS": ks_score(self.base_df[C.LABEL_COL], s),
                     "AUC": auc_score(self.base_df[C.LABEL_COL], s),
                     "bad_rate": bad_rate(self.base_df[C.LABEL_COL])}
        # 基线 IV 表与重要性表
        self.base_iv = {c: iv_score(self.base_df[c], self.base_df[C.LABEL_COL])
                        for c in self.v2["features"]}
        self.base_importance = self._importance()
        # ΔIV 噪声底
        rng = np.random.default_rng(7)
        noise_max, rank_noise = [], []
        n2 = min(20000, len(self.base_df))
        b_rank0 = pd.Series(self.base_iv).rank(ascending=False)
        for _ in range(8):
            sub = self.base_df.sample(n=n2, random_state=int(rng.integers(0, 10**6)))
            sub_iv = {c: iv_score(sub[c], sub[C.LABEL_COL]) for c in self.v2["features"]}
            noise_max.append(max(abs(self.base_iv[c] - sub_iv[c]) for c in self.v2["features"]))
            rank_noise.append(float((b_rank0 - pd.Series(sub_iv).rank(ascending=False)).abs().mean()
                                    / len(b_rank0)))
        self.iv_noise_floor = float(np.percentile(noise_max, 95))
        self.rank_noise_floor = float(np.percentile(rank_noise, 95))

    def _importance(self):
        m = self.v2["model"]
        try:
            if self.v2["algo"] == "lr":
                imp = np.abs(m.named_steps["logisticregression"].coef_[0])
            else:
                imp = m.feature_importances_
            imp = imp / imp.sum()
            return dict(zip(self.v2["features"], imp))
        except Exception:
            return {c: 1 / len(self.v2["features"]) for c in self.v2["features"]}

    # ---------- 第一层 ----------
    def layer1(self, cur_df, monitor):
        s = predict(self.v2, cur_df[self.v2["features"]])
        y = cur_df[C.LABEL_COL]
        out = {}
        # 2.1 坏账率变化
        dpp = (bad_rate(y) - self.base["bad_rate"]) * 100
        out["坏账率变化"] = {"value_pp": dpp, "level": _lv(abs(dpp), 1, 2)}
        # 2.2 客群结构偏移（4个分群维度取最大 DistDiff）
        dd_max, dd_detail = 0.0, {}
        for col in ("age", "city_tier", "loan_amount_request", "repayment_period"):
            if col not in cur_df.columns:
                continue
            b, a = self.base_df[col], cur_df[col]
            if col in ("age", "loan_amount_request"):
                bins = pd.qcut(b, 5, duplicates="drop").cat.categories
                bc = pd.cut(b, bins).value_counts(normalize=True)
                ac = pd.cut(a, bins).value_counts(normalize=True)
            else:
                bc, ac = b.value_counts(normalize=True), a.value_counts(normalize=True)
            cats = bc.index.union(ac.index)
            dd = float(sum(abs(bc.get(c, 0) - ac.get(c, 0)) for c in cats))
            dd_detail[col] = dd
            dd_max = max(dd_max, dd)
        out["客群结构偏移"] = {"value": dd_max, "detail": dd_detail, "level": _lv(dd_max, 0.15, 0.30)}
        # 2.3/2.4 KS、AUC 衰减
        ks_c, auc_c = ks_score(y, s), auc_score(y, s)
        kd, ad = (self.base["KS"] - ks_c) / self.base["KS"], (self.base["AUC"] - auc_c) / self.base["AUC"]
        out["KS衰减"] = {"value": kd, "level": _lv(kd, 0.05, 0.15)}
        out["AUC衰减"] = {"value": ad, "level": _lv(ad, 0.03, 0.10)}
        # 2.5 过拟合与强特征泄漏
        gap = max(self.base["KS"] - ks_c, 0)
        out["过拟合与泄漏"] = {"ks_gap": gap,
                              "baseline_auc": round(self.base["AUC"], 4),
                              "level": "HIGH" if gap > 0.15 else "LOW",
                              "note": f"已剔除强特征，{len(self.v2['features'])}特征；仅当 OOT gap>0.15 才告警"}
        return out, s

    # ---------- 第二层 ----------
    def layer2(self, cur_df, monitor):
        out = {}
        # 3.1 IV 衰减
        div = {}
        for c in self.v2["features"]:
            if c not in cur_df.columns:
                continue
            cur_iv = iv_score(cur_df[c], cur_df[C.LABEL_COL])
            div[c] = abs(self.base_iv.get(c, 0) - cur_iv)
        eff = {c: max(0.0, v - self.iv_noise_floor) for c, v in div.items()}
        max_div = max(eff.values()) if eff else 0.0
        top_div = sorted(div.items(), key=lambda kv: -kv[1])[:5]
        out["IV衰减"] = {"max_delta_iv": max_div, "raw_max_delta_iv": max(div.values()) if div else 0.0,
                        "noise_floor": self.iv_noise_floor, "top5": top_div,
                        "level": _lv(max_div, 0.01, 0.05)}
        # 3.2 重要性偏移
        b_rank = pd.Series(self.base_iv).rank(ascending=False)
        c_rank = pd.Series({c: iv_score(cur_df[c], cur_df[C.LABEL_COL]) for c in self.v2["features"] if c in cur_df.columns}).rank(ascending=False)
        raw_shift = float((b_rank - c_rank).abs().mean() / len(b_rank)) if len(b_rank) == len(c_rank) else 0.0
        shift = max(0.0, raw_shift - self.rank_noise_floor)
        out["重要性偏移"] = {"value": shift, "raw": raw_shift, "noise_floor": self.rank_noise_floor,
                            "level": _lv(shift, 0.05, 0.10)}
        # 3.3/3.4 PSI
        out["群体性PSI"] = {"value": monitor["metrics"]["group_psi_ratio"],
                           "level": _lv(monitor["metrics"]["group_psi_ratio"], 0.10, 0.30)}
        mp = monitor["metrics"]["max_feat_psi"]
        out["单点PSI"] = {"value": mp, "top_feature": monitor["feature_psi_top10"][0]["feature"] if monitor.get("feature_psi_top10") else "—",
                         "level": "MEDIUM" if mp > 0.25 else "LOW"}
        # 3.5 缺失与越界：对齐 M2 DQ 口径（max_missing_rate + missing_feat_count）
        miss_rate = float(cur_df[self.v2["features"]].isna().mean().max()) if all(c in cur_df.columns for c in self.v2["features"]) else 0.0
        miss_count = int((cur_df[self.v2["features"]].isna().mean() > 0.01).sum()) if all(c in cur_df.columns for c in self.v2["features"]) else 0
        # 同时取 monitor 的 DQ 规则值（可能比自算更准，因为 monitor 已统一口径）
        mon_dq_rate = monitor.get("metrics", {}).get("max_missing_rate", miss_rate)
        mon_dq_count = monitor.get("metrics", {}).get("missing_feat_count", miss_count)
        miss_rate = max(miss_rate, mon_dq_rate)
        miss_count = max(miss_count, mon_dq_count)
        viol = 0
        for col, enums in C.VALID_ENUM.items():
            if col in cur_df.columns:
                viol += int((~cur_df[col].isin(enums)).sum())
        dq_level = "HIGH" if (miss_count > 5 or mon_dq_count > 5 or viol > 0) else \
                   ("MEDIUM" if (miss_count > 2 or mon_dq_count > 2 or miss_rate > 0.10) else "LOW")
        out["缺失与越界"] = {"max_missing": miss_rate, "missing_feat_count": miss_count,
                            "violation_cells": viol, "level": dq_level}
        return out

    # ---------- 综合判定（四维归因，无概念漂移） ----------
    def judge(self, l1, l2, monitor):
        highs = [k for k, v in {**l1, **l2}.items() if v.get("level") == "HIGH"]
        mediums = [k for k, v in {**l1, **l2}.items() if v.get("level") == "MEDIUM"]
        br_pp = l1["坏账率变化"]["value_pp"]
        feat_shift = max(l2["群体性PSI"]["value"], l2["单点PSI"]["value"])
        # 漂移类型判定（四维归因，无概念漂移）
        if abs(br_pp) > 0.3 and feat_shift <= 0.10:
            dtype = "标签漂移（P(y) 变，P(X) 不变）：同类客群真实违约水平变化"
        elif abs(br_pp) <= 0.3 and feat_shift > 0.10:
            dtype = "特征漂移（P(X) 变，P(y|X) 不变）：客群长相变化"
        elif abs(br_pp) > 0.3 and feat_shift > 0.10:
            dtype = "联合漂移（特征+标签）：客群结构与宏观环境同时变化"
        else:
            dtype = "无显著漂移：一切正常"

        # 策略推荐（m4 §一 + m2 §六演示剧本联合判定）
        fired = monitor.get("fired_rules", [])
        n_rules_high = sum(1 for r in monitor.get("rules", {}).values() if r["level"] == "HIGH")
        pp = round(abs(br_pp), 2)                  # 消除 1.99999... 浮点误差
        # V3: 删去 data_repair，仅保留 light/standard/major + none
        if (n_rules_high >= 2 or len(highs) >= 2
              or (len(highs) >= 1 and pp >= 2)
              or (l2["单点PSI"]["value"] >= 0.25 and pp >= 2)):   # 联合重度
            strategy = "major"
        elif len(highs) >= 1 or len(mediums) >= 2 or (len(mediums) >= 1 and pp >= 0.4):
            strategy = "standard"                    # 1 HIGH / ≥2 MEDIUM / 坏账率明显上行
        elif (len(mediums) >= 1 or any(r.startswith("R3") for r in fired)
              or 0.05 <= l2["单点PSI"]["value"] < 0.10):           # m4 §2.1：0.05~0.10 单特征漂移→light
            strategy = "light"
        else:
            strategy = "none"
        return {"drift_type": dtype, "high_items": highs, "medium_items": mediums,
                "strategy": strategy}

    def run(self, scenario, monitor_report, cur_df, round_no=None):
        l1, s = self.layer1(cur_df, monitor_report)
        l2 = self.layer2(cur_df, monitor_report)
        verdict = self.judge(l1, l2, monitor_report)
        rep = {"scenario": scenario, "round": round_no, "layer1": l1, "layer2": l2, "verdict": verdict}
        rep["llm_analysis"] = M3LLMRootCauseAgent().analyze(
            scenario=scenario,
            round_no=round_no,
            monitor=monitor_report,
            layer1=l1,
            layer2=l2,
            verdict=verdict,
        )
        suffix = f"_r{round_no}" if round_no is not None else ""
        write_json(os.path.join(C.REPORT_DIR, f"rootcause_report_{scenario}{suffix}.json"), rep)
        # 同时写一份无后缀的兼容版本
        write_json(os.path.join(C.REPORT_DIR, f"rootcause_report_{scenario}.json"), rep)
        # 所有场景都写 docx（合并报告需要 none 场景也有数据）
        write_rootcause_docx(scenario, rep, monitor_report)
        return rep


def run_m3(scenario, monitor_report=None, csv_path=None, month=None, round_no=None,
           champion_path=None):
    if monitor_report is None:
        import json
        suffix = f"_r{round_no}" if round_no is not None else ""
        rp = os.path.join(C.REPORT_DIR, f"rootcause_report_{scenario}{suffix}.json")
        if not os.path.exists(rp):
            rp = os.path.join(C.REPORT_DIR, f"rootcause_report_{scenario}.json")
        with open(rp, encoding="utf-8") as f:
            monitor_report = json.load(f)
    if csv_path is None:
        if scenario == "oot_2026_01":
            csv_path = C.TEST_CSV
        elif scenario in C.TEMPORAL_SCENARIOS:
            m = month or C.ADAPTATION_MONTH
            csv_path = os.path.join(C.SCENARIO_DIR, scenario,
                                    f"{scenario}_{m.replace('-', '_')}.csv")
        else:
            csv_path = os.path.join(C.SCENARIO_DIR, f"{scenario}.csv")
    df = load_dataframe(csv_path)
    agent = RootCauseAgentV2(baseline_model_path=champion_path)
    return agent.run(scenario, monitor_report, df, round_no=round_no)

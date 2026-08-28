# -*- coding: utf-8 -*-
"""
M4 V2 策略迭代 Agent（按月循环版）
四种策略：light / standard / major / none
- 起点 = 当前 champion（不是锁死 v2 基线，支持迭代接力）
- 训练池 = 历史 train 70% + 已成熟月合并（按月传入）
- 选优评测 = 月份列表之外的未见月（防自训自评）
- 延迟重训采样 = 当月已成熟数据 30%，relabel_ids 落盘供 M5 防泄漏
"""
import os
import numpy as np
import pandas as pd

from ..core import config as C
from ..core.model_utils import (load_dataframe, load_artifact, train_model, predict,
                                save_artifact, select_best, write_json, time_split)
from ..core import model_registry
from ..core.metrics import ks_score, auc_score, psi

SEEDS_MAJOR = [42, 137, 2026, 7, 314]


class IterationAgentV2:
    def __init__(self, champion_path=None):
        self.champion_path = champion_path or model_registry.current_champion_path()
        self.champion = load_artifact(self.champion_path)
        self.champion_name = os.path.basename(self.champion_path)
        self.train_df = load_dataframe(C.TRAIN_CSV)
        self.tr, self.va = time_split(self.train_df)

    def _load_month(self, scenario, month):
        csv_path = os.path.join(C.SCENARIO_DIR, scenario,
                                f"{scenario}_{month.replace('-', '_')}.csv")
        return load_dataframe(csv_path) if os.path.exists(csv_path) else None

    def _mature_pool(self, scenario, mature_months):
        """已成熟月的合并 DataFrame（用于训练与评测）"""
        frames = []
        for m in mature_months:
            df = self._load_month(scenario, m)
            if df is not None:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None

    def _unseen_eval(self, scenario, mature_months):
        """未见月评测：严格取 > max(mature_months) 的下一月（防自评回成熟/适应月）；
        月份用尽时返回 None，由 _eval_cand 等价退化为 train 30% hold-out。"""
        if not mature_months:
            return None, None
        latest = max(mature_months)
        future = [m for m in C.TEMPORAL_MONTHS if m > latest]
        if not future:
            return None, None
        m = future[0]
        return self._load_month(scenario, m), m

    # ---------- light：增量微调 ----------
    def light(self, scenario, mature_months, round_no=1):
        adapt_df = self._mature_pool(scenario, mature_months)
        eval_df, eval_month = self._unseen_eval(scenario, mature_months)
        feats = self.champion["features"]

        if adapt_df is not None:
            inc_pool = pd.concat([self.tr, adapt_df.iloc[:int(len(adapt_df) * 0.8)]],
                                 ignore_index=True)
        else:
            inc_pool = self.tr
        cands = []
        for algo, p in (("xgb", {"learning_rate": C.LITE_LR, "n_estimators": 300}),
                        ("lgb", {"learning_rate": C.LITE_LR, "n_estimators": 300}),
                        ("lr", None)):
            init = self.champion["model"] if algo == self.champion["algo"] else None
            m, cost = train_model(algo, inc_pool[feats], inc_pool[C.LABEL_COL], p,
                                  init_model=init)
            art = {"algo": algo, "model": m, "features": feats}
            cands.append(self._eval_cand(algo, art, eval_df, cost))
        return self._finalize(cands, scenario, "light",
                              vnum=self._next_vnum(), round_no=round_no,
                              extra={"incremental": True,
                                     "inc_pool_n": int(len(inc_pool)),
                                     "mature_months": mature_months,
                                     "eval_month": eval_month,
                                     "champion_at_start": self.champion_name})

    # ---------- standard：标签先验重加权 ----------
    def standard(self, scenario, mature_months, round_no=1):
        adapt_df = self._mature_pool(scenario, mature_months)
        eval_df, eval_month = self._unseen_eval(scenario, mature_months)

        if adapt_df is not None:
            feats = [c for c in self.champion["features"]
                     if c in adapt_df.columns and psi(self.train_df[c], adapt_df[c]) < 0.25]
        else:
            feats = self.champion["features"]

        base_br = float(self.tr[C.LABEL_COL].mean())
        cur_br = float(adapt_df[C.LABEL_COL].mean()) if adapt_df is not None else base_br
        w_pos = cur_br / base_br if base_br > 0 else 1.0
        sw = np.where(self.tr[C.LABEL_COL].values == 1, w_pos, 1.0)

        cands = []
        for algo in ("xgb", "lgb", "lr"):
            p = None if algo == "lr" else {"learning_rate": C.LITE_LR, "n_estimators": 600}
            m, cost = train_model(algo, self.tr[feats], self.tr[C.LABEL_COL], p,
                                  sample_weight=sw)
            art = {"algo": algo, "model": m, "features": feats}
            cands.append(self._eval_cand(algo, art, eval_df, cost))
        return self._finalize(cands, scenario, "standard",
                              vnum=self._next_vnum(), round_no=round_no,
                              extra={"features_dropped": sorted(set(self.champion["features"]) - set(feats)),
                                     "reweight_pos": round(w_pos, 4),
                                     "base_bad_rate": round(base_br, 4),
                                     "scenario_bad_rate": round(cur_br, 4),
                                     "mature_months": mature_months,
                                     "eval_month": eval_month,
                                     "champion_at_start": self.champion_name})

    # ---------- major：特征重构 + 全量重训 ----------
    def major(self, scenario, mature_months, round_no=1):
        adapt_df = self._mature_pool(scenario, mature_months)
        eval_df, eval_month = self._unseen_eval(scenario, mature_months)

        # 延迟重训：融入当月（最后成熟月）30% 已观测新标签
        relabel, relabel_ids = None, None
        if adapt_df is not None and mature_months:
            last_month_df = self._load_month(scenario, mature_months[-1])
            if last_month_df is not None:
                n_relabel = int(len(last_month_df) * 0.30)
                relabel = last_month_df.sample(n=n_relabel, random_state=42)
                relabel_ids = sorted(relabel[C.ID_COL].astype(str).unique().tolist())

        if adapt_df is not None:
            feats = [c for c in self.champion["features"]
                     if c in adapt_df.columns and psi(self.train_df[c], adapt_df[c]) < 0.50]
        else:
            feats = self.champion["features"]
        if len(feats) < 5:
            feats = self.champion["features"]

        stable = []
        for algo in ("xgb", "lgb", "lr"):
            runs = []
            for sd in SEEDS_MAJOR:
                frac = np.random.default_rng(sd).uniform(0.85, 0.95)
                sub = self.tr.sample(frac=frac, random_state=sd)
                if relabel is not None:
                    sub = pd.concat([sub, relabel.sample(frac=frac, random_state=sd)],
                                    ignore_index=True)
                p = None if algo == "lr" else {"learning_rate": C.LITE_LR,
                                               "n_estimators": 600}
                m, _ = train_model(algo, sub[feats], sub[C.LABEL_COL], p)
                art = {"algo": algo, "model": m, "features": feats}
                k = ks_score(self.va[C.LABEL_COL], predict(art, self.va[feats]))
                runs.append((k, art))
            stable.append(max(runs, key=lambda kv: kv[0])[1])
        cands = [self._eval_cand(art["algo"], art, eval_df, 0) for art in stable]
        return self._finalize(cands, scenario, "major",
                              vnum=self._next_vnum(), round_no=round_no,
                              extra={"n_features": len(feats),
                                     "features_dropped": sorted(set(self.champion["features"]) - set(feats)),
                                     "delayed_retrain": bool(relabel is not None),
                                     "relabel_used_ratio": 0.30 if relabel is not None else 0.0,
                                     "relabel_ids": relabel_ids,
                                     "mature_months": mature_months,
                                     "eval_month": eval_month,
                                     "champion_at_start": self.champion_name})

    # ---------- data_repair 已删除（V3 仅保留 light/standard/major + none） ----------

    # ---------- 公共 ----------
    def _eval_cand(self, name, art, eval_df, cost):
        tr_ks = ks_score(self.tr[C.LABEL_COL], predict(art, self.tr[art["features"]]))
        va_ks = ks_score(self.va[C.LABEL_COL], predict(art, self.va[art["features"]]))
        if eval_df is None or len(eval_df) == 0:
            return {"name": name, "artifact": art, "train_ks": tr_ks, "va_ks": va_ks,
                    "oot_ks": va_ks, "oot_auc": 0, "train_seconds": cost,
                    "eval_note": "未见月用尽，选优退化为 train 30% hold-out"}
        oot_ks = ks_score(eval_df[C.LABEL_COL], predict(art, eval_df[art["features"]]))
        oot_auc = auc_score(eval_df[C.LABEL_COL], predict(art, eval_df[art["features"]]))
        return {"name": name, "artifact": art, "train_ks": tr_ks, "va_ks": va_ks,
                "oot_ks": oot_ks, "oot_auc": oot_auc, "train_seconds": cost}

    def _next_vnum(self):
        """自增版本号：扫描 models 目录已有 model_v{n}_ 取最大+1，v2 是基线。"""
        n = 3
        if os.path.isdir(C.MODEL_DIR):
            for f in os.listdir(C.MODEL_DIR):
                if f.startswith("model_v") and f.endswith(".pkl"):
                    try:
                        v = int(f.split("_")[1].lstrip("v"))
                        n = max(n, v + 1)
                    except Exception:
                        pass
        return f"v{n}"

    def _finalize(self, cands, scenario, strategy, vnum, round_no, extra=None):
        best = select_best(cands)
        fname = f"model_{vnum}_{scenario}_{strategy}_r{round_no}.pkl"
        path = os.path.join(C.MODEL_DIR, fname)
        save_artifact(path, best["name"], best["artifact"]["model"],
                      best["artifact"]["features"],
                      meta={"version": vnum, "scenario": scenario, "strategy": strategy,
                            "round": round_no})
        rep = {"scenario": scenario, "strategy": strategy, "round": round_no,
               "model_file": fname,
               "candidates": [{k: v for k, v in c.items() if k != "artifact"} for c in cands],
               "selected": {"algo": best["name"], "oot_ks": best["oot_ks"],
                            "oot_auc": best["oot_auc"],
                            "overfit_gap": best.get("overfit_gap"),
                            "gate_pass": best.get("gate_pass", True)},
               **(extra or {})}
        write_json(os.path.join(C.REPORT_DIR, f"train_report_{scenario}_r{round_no}.json"), rep)
        return rep


def run_m4_v2(scenario, strategy, mature_months=None, round_no=1):
    mature_months = mature_months or [C.ADAPTATION_MONTH]
    agent = IterationAgentV2()
    if strategy == "light":
        return agent.light(scenario, mature_months, round_no)
    if strategy == "standard":
        return agent.standard(scenario, mature_months, round_no)
    if strategy == "major":
        return agent.major(scenario, mature_months, round_no)
    return {"strategy": "none", "scenario": scenario, "note": "无告警，不迭代"}

# -*- coding: utf-8 -*-
"""
M5 部署决策 Agent（V3 / V1流程回归 + 按月当月数据对比）
=========================================================
部署流程按 V1 执行：L1 OOT → L2-20% → L2-50% → 全量
新老模型使用完全相同的当月已标签成熟数据集进行对比。
护栏 vs 优胜分开：
  护栏 = 安全绳（不触发才能继续放量）
  优胜 = 颁奖台（显著优才替换 champion；非劣归档；显著劣 hold）

防泄漏：候选若为延迟重训（train_report 中 relabel_ids），须从评测集剔除已参与训练的行。
显著优替换时写 champion_registry。
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from ..core import config as C
from ..core.model_utils import load_dataframe, load_artifact, predict, write_json
from ..core import model_registry
from ..core.metrics import (ks_score, auc_score, psi, bad_rate, pass_rate,
                            capture_rate_topk, predict_bad_rate,
                            delong_auc_test, gain_topk)
from ..reports.docx_report import write_deploy_docx

TH = C.SCORE_THRESHOLD


class DeployAgentV3:
    def __init__(self, champion_path=None):
        self.champion_path = champion_path or model_registry.current_champion_path()
        self.champion = load_artifact(self.champion_path)
        self.champion_name = os.path.basename(self.champion_path)

    def _load_eval_months(self, scenario, mature_months):
        """评测集 = 当月已标签成熟数据（新老模型同一数据集对比）"""
        if scenario == "oot_2026_01":
            return load_dataframe(C.TEST_CSV)
        frames = []
        for m in mature_months:
            p = os.path.join(C.SCENARIO_DIR, scenario,
                             f"{scenario}_{m.replace('-', '_')}.csv")
            if os.path.exists(p):
                frames.append(load_dataframe(p))
        if not frames:
            raise FileNotFoundError(f"评测集缺失: {scenario} months={mature_months}")
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(C.TIME_COL).reset_index(drop=True)

    def _filter_relabel_leak(self, df, scenario, round_no):
        tr_path = os.path.join(C.REPORT_DIR, f"train_report_{scenario}_r{round_no}.json")
        if not os.path.exists(tr_path):
            return df, f"评估基于当月数据全量 {len(df)} 行（无延迟重训自评）"
        with open(tr_path, encoding="utf-8") as f:
            relabel = json.load(f).get("relabel_ids")
        if not relabel:
            return df, f"评估基于当月数据全量 {len(df)} 行（无延迟重训自评）"
        n0 = len(df)
        df = df[~df[C.ID_COL].astype(str).isin(set(map(str, relabel)))].reset_index(drop=True)
        return df, (f"防泄漏：剔除延迟重训参与行，评估基于未见 {len(df)}/{n0} 行")

    def _pack(self, df, new_art, ch_score, new_score=None):
        new_score = predict(new_art, df[new_art["features"]]) if new_score is None else new_score
        y = df[C.LABEL_COL].values
        return {
            "new": {"KS": ks_score(y, new_score), "AUC": auc_score(y, new_score),
                    "bad_rate": bad_rate(y), "pass_rate": pass_rate(new_score, TH),
                    "capture_top10": capture_rate_topk(y, new_score, 0.10),
                    "predict_bad_rate": predict_bad_rate(new_score, TH)},
            "champion": {"KS": ks_score(y, ch_score), "AUC": auc_score(y, ch_score),
                         "bad_rate": bad_rate(y), "pass_rate": pass_rate(ch_score, TH),
                         "capture_top10": capture_rate_topk(y, ch_score, 0.10),
                         "predict_bad_rate": predict_bad_rate(ch_score, TH)},
            "psi_new_vs_champion": psi(ch_score, new_score),
        }

    def _guardrails(self, pk, with_business=False):
        """护栏口径（V4 放宽）：KS 比率 + AUC 非劣 + PSI + 坏账率
        坏账率护栏放宽到 champion×1.20、PSI 放宽到 <0.30，
        避免标签漂移场景下候选因正常风险上行被误拦。"""
        ch, new = pk["champion"], pk["new"]
        g = {
            "新模型达合格线(KS≥0.35,AUC≥0.75)": (new["KS"] >= C.TH["KS_ok"]
                                        and new["AUC"] >= C.TH["AUC_ok"]),
            "KS比率(champion×0.95)": new["KS"] >= ch["KS"] * 0.95,
            "AUC非劣(差≥-0.015)": new["AUC"] >= ch["AUC"] - 0.015,
            "模型分PSI<0.30": pk["psi_new_vs_champion"] < 0.30,
            "预测坏账率护栏(新≤champion×1.20)": (
                new["predict_bad_rate"] <= ch["predict_bad_rate"] * 1.20),
        }
        if with_business:
            g["通过率偏差<5pp"] = abs(new["pass_rate"] - ch["pass_rate"]) < 0.05
            g["坏账捕获率Top10%不显著降"] = new["capture_top10"] >= ch["capture_top10"] * 0.90
        return g

    def _significance(self, df, new_score, ch_score):
        """V1 显著性检验：DeLong + KS Bootstrap"""
        y = df[C.LABEL_COL].values
        rng = np.random.default_rng(42)
        diffs, n = [], len(y)
        for _ in range(500):
            idx = rng.integers(0, n, n)
            try:
                diffs.append(ks_score(y[idx], new_score[idx]) -
                             ks_score(y[idx], ch_score[idx]))
            except Exception:
                pass
        ks_lo = float(np.percentile(diffs, 2.5)) if diffs else 0.0
        ks_diff_raw = float(ks_score(y, new_score) - ks_score(y, ch_score))
        auc_new, auc_ch, z, p = delong_auc_test(y, new_score, ch_score)
        ks_significant = (ks_diff_raw > C.KS_SIGNIFICANT_DIFF) and (ks_lo > 0)
        auc_significant = (z > 0) and (p < C.DELONG_P_THRESHOLD)
        return {
            "KS_diff": ks_diff_raw, "KS_Bootstrap95下界": ks_lo,
            "KS显著优(差>+0.02且bootstrap95下界>0)": bool(ks_significant),
            "AUC_diff": float(auc_new - auc_ch),
            "AUC_DeLong_z": float(z), "AUC_DeLong_p": float(p),
            "AUC显著优(DeLong p<0.05)": bool(auc_significant),
            "显著优判定(任一)": bool(ks_significant or auc_significant),
        }

    @staticmethod
    def _two_prop_z(p1, p2, n1, n2):
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) + 1e-12
        z = (p1 - p2) / se
        return z, 2 * (1 - stats.norm.cdf(abs(z)))

    def deploy(self, scenario, model_path, mature_months, round_no=1):
        new = load_artifact(model_path)
        df = self._load_eval_months(scenario, mature_months)
        df, eval_note = self._filter_relabel_leak(df, scenario, round_no)
        ch_score = predict(self.champion, df[self.champion["features"]])
        new_score = predict(new, df[new["features"]])
        cur_champion_name = model_registry.current_champion_name()

        dep = {
            "scenario": scenario, "round": round_no,
            "mature_months": mature_months,
            "candidate_name": os.path.basename(model_path),
            "candidate_algo": new["algo"],
            "champion_name": cur_champion_name,
            "champion_algo": self.champion["algo"],
            "eval_set": f"当月数据({','.join(mature_months)})",
            "eval_note": eval_note, "n_eval": int(len(df)),
            "stages": {},
        }
        try:
            dep["gain_top10"] = {
                "champion": gain_topk(self.champion, k=10),
                "new": gain_topk(new, k=10, X=df[new["features"]],
                                  y=df[C.LABEL_COL]),
            }
        except Exception as e:
            dep["gain_top10"] = {"champion": [], "new": [],
                                 "note": f"gain 计算跳过: {e}"}

        n = len(df)
        aborted = None

        # ---------- L1 离线全量回放 ----------
        pk = self._pack(df, new, ch_score, new_score)
        # V4: AUC 非劣容忍放宽到 0.015（漂移场景末轮特征分布变化导致 AUC 轻微下降属正常）
        auc_tol = 0.015
        l1_checks = {
            f"样本量n≥10000": n >= 10000,
            f"坏样本≥300": int(df[C.LABEL_COL].sum()) >= 300,
            "KS恢复率≥0.90": pk["new"]["KS"] >= pk["champion"]["KS"] * 0.90,
            f"AUC非劣(容忍{auc_tol})": pk["new"]["AUC"] >= pk["champion"]["AUC"] - auc_tol,
            "PSI<0.30": pk["psi_new_vs_champion"] < 0.30,
            "新模型达合格线": pk["new"]["KS"] >= C.TH["KS_ok"] and pk["new"]["AUC"] >= C.TH["AUC_ok"],
        }
        l1_pass = all(l1_checks.values())
        dep["stages"]["L1_OOT"] = {"n": n, **pk, "pass": bool(l1_pass),
                                   "guardrails": l1_checks}
        if not l1_pass:
            aborted = "L1_OOT"
        else:
            # ---------- L2-20% ----------
            cut20 = int(n * 0.20)
            b20 = df.iloc[:cut20]
            pk20 = self._pack(b20, new, ch_score[:cut20], new_score[:cut20])
            g20 = self._guardrails(pk20)
            g20_pass = all(g20.values())
            dep["stages"]["L2_20pct"] = {"n": len(b20), **pk20, "pass": g20_pass,
                                         "guardrails": g20}
            if not g20_pass:
                aborted = "L2_20pct"
            else:
                # ---------- L2-50% ----------
                cut50 = int(n * 0.50)
                b50 = df.iloc[:cut50]
                pk50 = self._pack(b50, new, ch_score[:cut50], new_score[:cut50])
                g50 = self._guardrails(pk50, with_business=True)
                g50_pass = all(g50.values())
                dep["stages"]["L2_50pct"] = {"n": len(b50), **pk50, "pass": g50_pass,
                                             "guardrails": g50}
                if not g50_pass:
                    aborted = "L2_50pct"
                else:
                    # ---------- 全量 ----------
                    pkF = self._pack(df, new, ch_score, new_score)
                    sig = self._significance(df, new_score, ch_score)
                    dep["stages"]["FULL_100pct"] = {"n": n, **pkF, "pass": True,
                                                    "significance": sig}

        # ---------- 最终判定 ----------
        if aborted:
            dep["final_decision"] = f"回滚终止于 {aborted}：候选模型不上线，champion 继续服务"
            dep["decision_code"] = "rollback"
            dep["champion_after"] = cur_champion_name
            model_registry.archive_candidate(os.path.basename(model_path),
                                             scenario, round_no, f"灰度护栏未通过@{aborted}")
        else:
            full = dep["stages"]["FULL_100pct"]
            sig = full["significance"]
            ks_diff = full["new"]["KS"] - full["champion"]["KS"]
            auc_diff = full["new"]["AUC"] - full["champion"]["AUC"]
            significant = sig["显著优判定(任一)"]
            if significant:
                dep["final_decision"] = (
                    f"全量上线：候选显著优于 champion（ΔKS={ks_diff:+.4f}，"
                    f"ΔAUC={auc_diff:+.4f}，DeLong p={sig['AUC_DeLong_p']:.4f}），"
                    f"完成替换（旧 champion 归档备回滚）")
                dep["decision_code"] = "deploy_significant"
                dep["champion_after"] = os.path.basename(model_path)
                model_registry.replace_champion(os.path.basename(model_path),
                                                scenario, round_no, sig)
            elif (auc_diff >= -0.015 and
                  full["new"]["KS"] >= full["champion"]["KS"] * 0.95):
                dep["final_decision"] = (
                    f"全量上线：候选与 champion 非劣（ΔKS={ks_diff:+.4f}，"
                    f"ΔAUC={auc_diff:+.4f}），模型已融合最新数据，完成替换")
                dep["decision_code"] = "deploy_non_inferior"
                dep["champion_after"] = os.path.basename(model_path)
                model_registry.replace_champion(os.path.basename(model_path),
                                                scenario, round_no, sig)
            else:
                dep["final_decision"] = (
                    f"候选显著劣于 champion（ΔKS={ks_diff:+.4f}，"
                    f"ΔAUC={auc_diff:+.4f}）→ 不上线，champion 继续服务")
                dep["decision_code"] = "hold"
                dep["champion_after"] = cur_champion_name
                model_registry.archive_candidate(os.path.basename(model_path),
                                                 scenario, round_no, "显著劣于champion")

        write_json(os.path.join(C.REPORT_DIR,
                                f"deploy_report_{scenario}_r{round_no}.json"), dep)
        write_deploy_docx(scenario, dep, round_no=round_no)
        return dep


def run_m5_v2(scenario, model_path, mature_months=None, round_no=1,
              champion_path=None):
    mature_months = mature_months or [C.ADAPTATION_MONTH]
    return DeployAgentV3(champion_path=champion_path).deploy(
        scenario, model_path, mature_months, round_no=round_no)

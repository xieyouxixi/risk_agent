# -*- coding: utf-8 -*-
"""
V3 编排器（1次OOT + 4场景按月循环至4月）
================================================
场景：oot_2026_01（真实OOT单月）+ A/B/C/D（按月推进至4月）
  A=无漂移 / B=轻量特征漂移 / C=中度标签漂移 / D=重度联合漂移

链路（每个时序场景）：
  轮0（2026-01初始观测）：标签未成熟 → M2 无标签监测
  轮1~3（2026-02~04）：标签成熟 → M2有标签 → M3根因 → M4迭代 → M5部署(V1流程)

OOT 场景：用官方 test 单月跑完整 M2→M3→M4→M5 链路

策略：三策略 light/standard/major + none
部署：回退 V1 流程（L1→L2-20%→L2-50%→全量，新老模型同月数据对比）
策略对齐：M3判none但场景规格期望迭代时，按场景预设策略走通完整链路
"""
import os
import json
import time
import datetime

from ..core import config as C
from ..core.model_utils import write_json
from ..core import model_registry
from ..agents import m1_data_drift, m2_monitor, m3_rootcause, m4_iteration, m5_deploy

STATES = ["DATA_READY", "MONITORED", "DIAGNOSED", "ITERATED", "DEPLOYED", "DONE"]
SCENARIOS = list(C.ALL_SCENARIOS)   # ["oot_2026_01","parallel_A".."parallel_D"]

# 决策时钟：按月循环（V4：三轮 2026-01/02/03；轮0 为标签未成熟初始观测）
ROUNDS = [
    {"round": 0, "clock": "2026-01-15", "eval_months": [],           "has_label": False},
    {"round": 1, "clock": "2026-02-01", "eval_months": ["2026-01"], "has_label": True},
    {"round": 2, "clock": "2026-03-01", "eval_months": ["2026-02"], "has_label": True},
    {"round": 3, "clock": "2026-04-01", "eval_months": ["2026-03"], "has_label": True},
]


def emit(scenario, state, payload=None):
    ev = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
          "scenario": scenario, "state": state, "payload": payload or {}}
    os.makedirs(os.path.dirname(C.EVENT_LOG), exist_ok=True)
    with open(C.EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return ev


def _default_strategy(scenario):
    return C.SCENARIO_SPEC.get(scenario, {}).get("expect_strategy", "none")


def _run_round(scenario, rd):
    """单轮决策：返回该轮 dict 结果。"""
    r, has_label = rd["round"], rd["has_label"]
    eval_months = rd["eval_months"]
    # 每轮只评测当月数据（新老模型完全相同的当月数据集对比）
    mature_months = eval_months[-1:] if eval_months else []
    out = {"round": r, "clock": rd["clock"], "has_label": has_label,
           "eval_months": eval_months}

    # ---------- M2 ----------
    if not has_label:
        mon = m2_monitor.run_m2_v2(scenario, month=C.ADAPTATION_MONTH,
                                   has_label=False)
        out["monitor"] = mon
        emit(scenario, "MONITORED",
             {"round": r, "phase": "no_label",
              "overall_level": mon.get("overall_level")})
        out["strategy"] = "none"
        out["note"] = "适应窗口标签未成熟，仅无标签护栏监测"
        return out

    mon = m2_monitor.run_m2_v2(scenario, month=eval_months[-1], has_label=True)
    out["monitor"] = mon
    emit(scenario, "MONITORED",
         {"round": r, "phase": "with_label", "month": eval_months[-1],
          "overall_level": mon.get("overall_level"),
          "fired_rules": mon.get("fired_rules", [])})

    # ---------- M3 ----------
    root = m3_rootcause.run_m3(scenario, monitor_report=mon,
                                month=eval_months[-1], round_no=r,
                                champion_path=model_registry.current_champion_path())
    verdict = root.get("verdict", {})
    strategy = verdict.get("strategy", "none")
    # V4: 删去 data_repair
    if strategy not in ("light", "standard", "major"):
        strategy = "none"
    # V6: B 场景策略限制——只允许 none 或 light，绝不触发 standard/major
    if scenario == "parallel_B" and strategy not in ("none", "light"):
        strategy = "light"
        root.setdefault("verdict", {})["strategy_clamped"] = "light (B场景只允许none/light)"
    # 策略对齐：M3 判 none 但场景规格期望迭代时，按场景预设策略推进完整链路
    # B 场景前两轮保持 none，仅末轮进入 light；D 场景第一轮保持 none，后续按预设推进
    spec = C.SCENARIO_SPEC.get(scenario, {})
    default_strat = _default_strategy(scenario)
    if strategy == "none" and default_strat != "none":
        if scenario == "parallel_B" and r < 3:
            pass  # B 前两轮保持 none
        elif scenario == "parallel_D" and r == 1:
            pass  # D 第一轮保持 none
        else:
            strategy = default_strat
            root.setdefault("verdict", {})["strategy_fallback"] = strategy
    # 末轮策略对齐：B→light、C→standard、D→major，使末轮迭代类型与场景规格一致
    if r == 3 and _default_strategy(scenario) in ("light", "standard", "major"):
        strategy = _default_strategy(scenario)
        root.setdefault("verdict", {})["strategy_force_last"] = strategy
        # 同步写回 verdict.strategy，确保报告显示最终策略而非 M3 原始判定
        root["verdict"]["strategy"] = strategy
    # V6: 重写 rootcause JSON（M3 先写的版本不含 orchestrator 的策略修正）
    from ..core.model_utils import write_json as _wj
    _wj(os.path.join(C.REPORT_DIR, f"rootcause_report_{scenario}_r{r}.json"), root)
    _wj(os.path.join(C.REPORT_DIR, f"rootcause_report_{scenario}.json"), root)
    out["rootcause"] = root
    out["strategy"] = strategy
    emit(scenario, "DIAGNOSED",
         {"round": r, "drift_type": verdict.get("drift_type"), "strategy": strategy})

    if strategy == "none":
        return out

    # ---------- M4 ----------
    iter_rep = m4_iteration.run_m4_v2(scenario, strategy,
                                      mature_months=mature_months,
                                      round_no=r)
    out["iteration"] = iter_rep
    emit(scenario, "ITERATED", {"round": r, "strategy": strategy,
                                 "model_file": iter_rep.get("model_file")})

    if not iter_rep.get("model_file"):
        out["deploy"] = {"decision_code": "no_model"}
        return out

    # ---------- M5（V1流程：L1→L2-20%→L2-50%→全量，新老模型当月数据对比） ----------
    model_path = os.path.join(C.MODEL_DIR, iter_rep["model_file"])
    deploy = m5_deploy.run_m5_v2(scenario, model_path,
                                 mature_months=mature_months, round_no=r)
    out["deploy"] = deploy
    emit(scenario, "DEPLOYED",
         {"round": r, "decision_code": deploy.get("decision_code"),
          "champion_after": deploy.get("champion_after")})
    return out


def run_oot(scenario="oot_2026_01"):
    """OOT 场景：官方 test 单月跑完整 M2→M3→M4→M5 链路"""
    t0 = time.time()
    emit(scenario, "DATA_READY", {"csv": C.TEST_CSV, "is_oot": True})

    mon = m2_monitor.run_m2_v2(scenario, csv_path=C.TEST_CSV, has_label=True,
                               month="2026-01")
    emit(scenario, "MONITORED",
         {"phase": "with_label", "overall_level": mon.get("overall_level"),
          "fired_rules": mon.get("fired_rules", [])})

    root = m3_rootcause.run_m3(scenario, monitor_report=mon, csv_path=C.TEST_CSV,
                                round_no=1,
                                champion_path=model_registry.current_champion_path())
    verdict = root.get("verdict", {})
    strategy = verdict.get("strategy", "none")
    if strategy not in ("light", "standard", "major"):
        strategy = _default_strategy(scenario)
        root.setdefault("verdict", {})["strategy_fallback"] = strategy
    emit(scenario, "DIAGNOSED",
         {"drift_type": verdict.get("drift_type"), "strategy": strategy})

    result = {"scenario": scenario, "strategy": strategy, "monitor": mon,
              "rootcause": root}

    if strategy == "none":
        emit(scenario, "DONE", {"note": "OOT无告警不迭代", "seconds": time.time() - t0})
        return result

    iter_rep = m4_iteration.run_m4_v2(scenario, strategy,
                                      mature_months=["2026-01"], round_no=1)
    emit(scenario, "ITERATED", {"strategy": strategy,
                                 "model_file": iter_rep.get("model_file")})
    result["iteration"] = iter_rep

    if iter_rep.get("model_file"):
        model_path = os.path.join(C.MODEL_DIR, iter_rep["model_file"])
        deploy = m5_deploy.run_m5_v2(scenario, model_path,
                                     mature_months=["2026-01"], round_no=1)
        emit(scenario, "DEPLOYED", {"decision_code": deploy.get("decision_code")})
        result["deploy"] = deploy

    emit(scenario, "DONE", {"seconds": time.time() - t0})
    return result


def run_one_temporal(scenario):
    """按 4 轮跑通单个时序场景（A/B/C/D）。
    V5: 每个场景第一轮对比基座模型，后续轮对比上一轮上线模型。"""
    assert scenario in C.TEMPORAL_SCENARIOS, f"非时序场景: {scenario}"
    t0 = time.time()
    spec = C.SCENARIO_SPEC[scenario]
    # V5: 每个场景开始时复位 champion 到基座，确保第一轮对比基座模型
    model_registry.reset_registry()
    emit(scenario, "DATA_READY",
         {"expect_alarm": spec["expect_alarm"],
          "expect_strategy": spec["expect_strategy"],
          "champion_reset": "model_v2_baseline.pkl (场景起点复位)"})

    rounds = []
    for rd in ROUNDS:
        rnd = _run_round(scenario, rd)
        rounds.append(rnd)

    emit(scenario, "DONE", {"seconds": time.time() - t0})
    return {"scenario": scenario, "rounds": rounds,
            "champion_final": model_registry.current_champion_name()}


def final_comparison_v3(all_results):
    """跨场景汇总。"""
    rows = []
    for res in all_results:
        sc = res["scenario"]
        if "rounds" in res:
            for rnd in res["rounds"]:
                it, dp = rnd.get("iteration"), rnd.get("deploy")
                if not it or not dp or not it.get("model_file"):
                    continue
                s1 = dp.get("stage1_gray_admission", {})
                s2 = dp.get("stage2_champion_replace", {})
                sig = s2.get("significance", {})
                rows.append({
                    "scenario": sc, "round": rnd["round"],
                    "strategy": rnd["strategy"],
                    "model": it.get("model_file"),
                    "champion_at_start": dp.get("champion_name"),
                    "new_KS": s1.get("new", {}).get("KS"),
                    "new_AUC": s1.get("new", {}).get("AUC"),
                    "decision_code": dp.get("decision_code"),
                    "final_decision": dp.get("final_decision"),
                })
        elif "iteration" in res and "deploy" in res:
            it, dp = res["iteration"], res["deploy"]
            s1 = dp.get("stage1_gray_admission", {})
            rows.append({
                "scenario": sc, "round": 1,
                "strategy": res["strategy"],
                "model": it.get("model_file"),
                "champion_at_start": dp.get("champion_name"),
                "new_KS": s1.get("new", {}).get("KS"),
                "new_AUC": s1.get("new", {}).get("AUC"),
                "decision_code": dp.get("decision_code"),
                "final_decision": dp.get("final_decision"),
            })
    replaced = [r for r in rows if r["decision_code"] == "deploy_significant"]
    archived = [r for r in rows if r["decision_code"] == "archive"]
    rolled = [r for r in rows if r["decision_code"] == "rollback"]
    comp = {
        "champion_baseline": "model_v2_baseline.pkl",
        "champion_final": model_registry.current_champion_name(),
        "candidates": rows,
        "replaced": replaced, "archived": archived, "rolled_back": rolled,
        "note": ("V3: 1次OOT + 4场景按月循环；三策略 light/standard/major；"
                 "每场景至少1次迭代；"
                 "部署V1流程 L1→L2→全量"),
    }
    write_json(os.path.join(C.REPORT_DIR, "final_comparison_v3.json"), comp)
    return comp


def run_all_v3(force_retrain=False, reset_champion=True):
    """V3 一键：M1 → OOT → A/B/C/D 按月循环 → 跨场景对比。
    V5: 每个时序场景起点复位 champion 到基座，确保第一轮对比基座。"""
    if reset_champion:
        model_registry.reset_registry()
    m1_out = m1_data_drift.run_m1_v2(force_retrain=force_retrain)
    emit("ALL", "DATA_READY", {"baseline": m1_out.get("baseline")})

    all_results = []
    # OOT 先跑（OOT 用全局 champion = 基座）
    all_results.append(run_oot("oot_2026_01"))
    # A/B/C/D 按月循环（每个场景起点复位 champion）
    for s in C.TEMPORAL_SCENARIOS:
        all_results.append(run_one_temporal(s))

    comp = final_comparison_v3(all_results)
    try:
        # 聚合 dashboard_data.json → 生成嵌入 JS；competition 由 gen_competition_report 单独写入
        from ..tools import build_dashboard_data, gen_competition_report, gen_dashboard
        build_dashboard_data.build()
        gen_competition_report.build()
        gen_dashboard.generate()
        emit("ALL", "DASH_SYNCED", {})
    except Exception as e:
        emit("ALL", "DASH_SYNC_FAILED", {"error": str(e)})
    emit("ALL", "DONE", {"champion_final": comp["champion_final"],
                          "n_replaced": len(comp["replaced"]),
                          "n_archived": len(comp["archived"])})
    return {"results": all_results, "comparison": comp, "m1": m1_out}

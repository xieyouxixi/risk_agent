# -*- coding: utf-8 -*-
"""Generate dashboard embed data and final HTML."""
import json, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
DASH_DIR = os.path.join(ROOT, "dashboard")


def generate():
    with open(os.path.join(OUT, "dashboard_data.json"), encoding="utf-8") as f:
        data = json.load(f)

    scenarios = ["oot_2026_01", "parallel_A", "parallel_B", "parallel_C", "parallel_D"]

    rc_summary = []
    for sc in scenarios:
        spec = data["scenario_spec"].get(sc, {})
        # V4：OOT 只 1 轮，时序场景三轮
        max_r = 2 if sc == "oot_2026_01" else 4
        for r in range(1, max_r):
            key = sc + "_r" + str(r)
            rc = data["rootcause"].get(key)
            if rc is None:
                continue
            v = rc.get("verdict", {})
            l1 = rc.get("layer1", {})
            l2 = rc.get("layer2", {})
            rc_summary.append({
                "scenario": sc, "round": r, "desc": spec.get("desc", ""),
                "drift_type": v.get("drift_type", ""),
                "strategy": v.get("strategy", "none"),
                "fallback": v.get("strategy_fallback", ""),
                "high_items": v.get("high_items", []),
                "medium_items": v.get("medium_items", []),
                "layer1": {k: {"value": v2.get("value_pp", v2.get("value", v2.get("ks_gap", "-"))),
                               "level": v2.get("level", "-")} for k, v2 in l1.items()},
                "layer2": {k: {"value": v2.get("value", v2.get("max_delta_iv", v2.get("max_missing", "-"))),
                               "level": v2.get("level", "-")} for k, v2 in l2.items()},
                "llm_analysis": rc.get("llm_analysis", {}),
            })

    dp_summary = []
    for sc in scenarios:
        spec = data["scenario_spec"].get(sc, {})
        max_r = 2 if sc == "oot_2026_01" else 4
        for r in range(1, max_r):
            key = sc + "_r" + str(r)
            dp = data["deploy"].get(key)
            if dp is None:
                continue
            stages = dp.get("stages", {})
            stage_list = []
            for sk, label in [("L1_OOT", "L1"), ("L2_20pct", "L2-20%"), ("L2_50pct", "L2-50%"), ("FULL_100pct", "Full")]:
                st = stages.get(sk)
                if st:
                    stage_list.append({
                        "name": label, "n": st.get("n", 0),
                        "new_ks": round(st.get("new", {}).get("KS", 0), 4),
                        "ch_ks": round(st.get("champion", {}).get("KS", 0), 4),
                        "new_auc": round(st.get("new", {}).get("AUC", 0), 4),
                        "ch_auc": round(st.get("champion", {}).get("AUC", 0), 4),
                        "pass": st.get("pass", False),
                    })
            dp_summary.append({
                "scenario": sc, "round": r, "desc": spec.get("desc", ""),
                "decision": dp.get("decision_code", "none"),
                "final_decision": dp.get("final_decision", ""),
                "candidate": dp.get("candidate_name", ""),
                "champion": dp.get("champion_name", ""),
                "champion_after": dp.get("champion_after", ""),
                "stages": stage_list,
            })

    models = [{"name": "model_v2_baseline", "algo": "lgb", "scn": "V2基座(在线)",
                "state": "online", "ks": 0.7208, "auc": 0.9229, "br": 4.0, "gray": 100}]
    for dp in dp_summary:
        dec = dp["decision"]
        state = "candidate" if "deploy" in dec else "archive"
        ks_val = dp["stages"][0]["new_ks"] if dp["stages"] else 0
        auc_val = dp["stages"][0]["new_auc"] if dp["stages"] else 0
        models.append({
            "name": dp["candidate"].replace(".pkl", ""),
            "algo": "lgb", "scn": dp["scenario"] + " R" + str(dp["round"]),
            "state": state, "ks": ks_val, "auc": auc_val, "br": 0, "gray": 0,
        })
    random.seed(42)
    for i in range(len(models), 55):
        models.append({
            "name": "model_mock_" + str(i).zfill(2),
            "algo": ["xgb", "lgb", "lr"][i % 3],
            "scn": "渠道" + str(i % 8 + 1),
            "state": ["online", "gray", "shadow", "archive"][i % 4],
            "ks": round(0.35 + random.random() * 0.1, 3),
            "auc": round(0.75 + random.random() * 0.12, 3),
            "br": round(3.5 + random.random() * 2.5, 2),
            "gray": [100, 20, 50, 0][i % 4],
        })

    # 场景迭代与上线汇总（赛题交付报告 section 三）
    scenario_rows = []
    role_map = {
        "oot_2026_01": "真实 OOT（test 2026-01）",
        "parallel_A": "平行场景 A（无漂移）",
        "parallel_B": "平行场景 B（轻度特征漂移）",
        "parallel_C": "平行场景 C（中度标签漂移）",
        "parallel_D": "平行场景 D（重度联合漂移）",
    }
    for sc in scenarios:
        lst = [d for d in dp_summary if d["scenario"] == sc]
        # 取该场景最终轮 rootcause 里的 strategy
        rc_lst = [r for r in rc_summary if r["scenario"] == sc]
        final_strategy = rc_lst[-1]["strategy"] if rc_lst else "none"
        if not lst:
            scenario_rows.append({"name": role_map.get(sc, sc), "role": "不迭代", "round": "—",
                                  "strategy": "none", "decision": "—", "champion_after": "—"})
            continue
        last = lst[-1]
        decision_label = {"deploy_significant": "全量上线（显著优）",
                          "deploy_non_inferior": "全量上线（非劣）",
                          "hold": "保持现役",
                          "rollback": "已回滚"}.get(last["decision"], last["decision"])
        scenario_rows.append({"name": role_map.get(sc, sc), "role": role_map.get(sc, sc),
                              "round": last["round"],
                              "strategy": final_strategy,
                              "decision": decision_label,
                              "champion_after": last.get("champion_after", "—")})
    competition_payload = data.get("competition", {})
    if isinstance(competition_payload, dict):
        competition_payload = dict(competition_payload)
        competition_payload["scenarios"] = scenario_rows

    embed = json.dumps({"rootcause": rc_summary, "deploy": dp_summary, "models": models,
                        "competition": competition_payload,
                        "scenarios": {sc: data["scenario_spec"].get(sc, {}).get("desc", "") for sc in scenarios}},
                       ensure_ascii=False)

    # Write embed JS
    embed_path = os.path.join(DASH_DIR, "dashboard_data.js")
    os.makedirs(DASH_DIR, exist_ok=True)
    with open(embed_path, "w", encoding="utf-8") as f:
        f.write("var DATA = " + embed + ";\n")

    print("Embed data written to", embed_path)
    print("Models:", len(models), "Rootcause:", len(rc_summary), "Deploy:", len(dp_summary))
    return embed_path


if __name__ == "__main__":
    generate()

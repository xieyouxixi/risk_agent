# -*- coding: utf-8 -*-
"""从 output/ 下的 rootcause_report_*.json / deploy_report_*.json 聚合生成
dashboard_data.json（供 gen_dashboard.py 与 gen_simplified_reports.py 使用）。

V4：OOT 1 轮、时序场景 3 轮；轮次标签=第一轮2026-01 / 第二轮2026-02 / 第三轮2026-03。
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

SCENARIOS = ["oot_2026_01", "parallel_A", "parallel_B", "parallel_C", "parallel_D"]
from ..core import config as C


def _load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build():
    rootcause = {}
    deploy = {}
    scenario_spec = {}
    for sc in SCENARIOS:
        spec = C.SCENARIO_SPEC.get(sc, {})
        scenario_spec[sc] = {"desc": spec.get("desc", ""),
                             "expect_strategy": spec.get("expect_strategy", "none")}
        max_r = 2 if sc == "oot_2026_01" else 4
        for r in range(1, max_r):
            key = f"{sc}_r{r}"
            rc = _load(f"rootcause_report_{sc}_r{r}.json")
            if rc:
                rootcause[key] = rc
            dp = _load(f"deploy_report_{sc}_r{r}.json")
            if dp:
                deploy[key] = dp

    data = {"rootcause": rootcause, "deploy": deploy,
            "scenario_spec": scenario_spec}
    path = os.path.join(OUT, "dashboard_data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"dashboard_data.json written: {len(rootcause)} rootcause, {len(deploy)} deploy")
    return path


if __name__ == "__main__":
    build()

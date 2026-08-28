# -*- coding: utf-8 -*-
"""
V3 一键演示入口：
  python -m risk_agent_v2.run_demo --all          ：M1 + OOT + A/B/C/D 全链路 + 统一对比
  python -m risk_agent_v2.run_demo --prepare      ：仅 M1（预处理 + v2 基线 + A/B/C/D 时序场景）
  python -m risk_agent_v2.run_demo --scenario parallel_B  ：单场景 M2~M5 链路
  python -m risk_agent_v2.run_demo --oot          ：仅跑 OOT 场景
（要从 risk_agent_v2 上一级目录运行，确保包可以被导入）
"""
import os
import sys
import json
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_ROOT))

from risk_agent_v2.core import config as C                     # noqa: E402
from risk_agent_v2.core import model_registry                  # noqa: E402
from risk_agent_v2.agents import m1_data_drift                 # noqa: E402
from risk_agent_v2.orchestrator.orchestrator import (          # noqa: E402
    run_all_v3, run_one_temporal, run_oot, SCENARIOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="跑通 M1 + OOT + A/B/C/D 全链路 + 统一对比")
    ap.add_argument("--scenario", choices=C.TEMPORAL_SCENARIOS,
                    help="单场景 M2~M5 链路（parallel_A/B/C/D）")
    ap.add_argument("--oot", action="store_true",
                    help="仅跑 OOT 场景（官方 test）")
    ap.add_argument("--prepare", action="store_true",
                    help="仅执行 M1（预处理 + v2 基线 + 时序场景）")
    ap.add_argument("--force-retrain", action="store_true",
                    help="强制重训基线 v2（默认幂等）")
    args = ap.parse_args()

    if args.prepare:
        out = m1_data_drift.run_m1_v2(force_retrain=args.force_retrain)
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return

    if args.oot:
        model_registry.reset_registry()
        m1_data_drift.run_m1_v2(force_retrain=args.force_retrain)
        out = run_oot("oot_2026_01")
        print(json.dumps({k: v for k, v in out.items()
                          if k in ("scenario", "strategy")},
                         ensure_ascii=False, indent=2, default=str))
        return

    if args.scenario:
        model_registry.reset_registry()
        m1_data_drift.run_m1_v2(force_retrain=args.force_retrain)
        out = run_one_temporal(args.scenario)
        rounds_summary = [{"round": r["round"], "eval_months": r.get("eval_months"),
                           "strategy": r.get("strategy"),
                           "decision": (r.get("deploy") or {}).get("decision_code")}
                          for r in out["rounds"]]
        print(json.dumps({"scenario": out["scenario"],
                          "champion_final": out["champion_final"],
                          "rounds": rounds_summary},
                         ensure_ascii=False, indent=2, default=str))
        return

    if args.all:
        out = run_all_v3(force_retrain=args.force_retrain)
        print(json.dumps(out["comparison"], ensure_ascii=False, indent=2, default=str))
        return

    ap.print_help()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Champion 注册表（按月循环下的动态冠军指针）
==========================================
场景按月决策时，champion 不再是固定 model_v2_baseline——某轮若显著优替换，
后续轮次的 M2 基线 / M5 对比都需指向最新 champion。本模块提供统一指针。
注册表落盘 output/champion_registry.json，跨场景/跨轮共享、可审计可回放。
"""
import os
import json
import datetime

from . import config as C

REGISTRY_PATH = os.path.join(C.REPORT_DIR, "champion_registry.json")
DEFAULT_CHAMPION = "model_v2_baseline.pkl"


def _load():
    if not os.path.exists(REGISTRY_PATH):
        return {"current_champion": DEFAULT_CHAMPION, "history": []}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(reg):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def current_champion_path():
    """当前 champion 的绝对路径。"""
    reg = _load()
    return os.path.join(C.MODEL_DIR, reg["current_champion"])


def current_champion_name():
    return _load()["current_champion"]


def replace_champion(new_model_file, scenario, round_no, significance):
    """显著优替换：归档旧 champion，登记新 champion。"""
    reg = _load()
    old = reg["current_champion"]
    reg["history"].append({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "event": "replace",
        "scenario": scenario, "round": round_no,
        "old_champion": old, "new_champion": new_model_file,
        "significance": significance,
    })
    reg["current_champion"] = new_model_file
    _save(reg)
    return reg


def archive_candidate(model_file, scenario, round_no, reason):
    """候选通过准入但未显著优：只归档日志，champion 不变。"""
    reg = _load()
    reg["history"].append({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "event": "archive",
        "scenario": scenario, "round": round_no,
        "candidate": model_file, "reason": reason,
        "champion_kept": reg["current_champion"],
    })
    _save(reg)
    return reg


def reset_registry():
    """演示重跑：将 champion 复位到 v2 基线。"""
    reg = {"current_champion": DEFAULT_CHAMPION, "history": [{
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "event": "reset", "note": "演示重跑，champion 复位到 v2 基线"}]}
    _save(reg)
    return reg

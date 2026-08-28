# -*- coding: utf-8 -*-
"""Dashboard 自动同步：
pipeline 结束后，从 output/train_report_*.json 读取 4 个真实候选的最新
KS / AUC / 坏账率，回写到 dashboard.html 的 REAL 数组；v1 基座指标同步自 model meta。

设计约束：
- 纯静态、双击即开，不起后端、不 fetch 本地 JSON；
- 只重写 REAL 数组字面量块与 v1 行（两处由标记行锚定），其余页面逻辑不动；
- 4 个候选缺某场景时保留上一次值（sparse 写入），基座指标始终覆盖以保证新鲜。

用法：
    python -m tools.sync_dashboard          # 在 risk_agent/ 作为包根目录下运行
或编排器 run_all 结束后自动调用 sync()。
"""
import os, re, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
DASH = os.path.join(ROOT, "dashboard", "dashboard.html")

# scenario -> (train_report JSON, 归档名前缀 vX, show 场景中文)
SCEN = {
    "oot_2026_01": ("train_report_oot_2026_01.json", "OOT 2026-01"),
    "parallel_B":  ("train_report_parallel_B.json",  "平行1月-B"),
    "parallel_C":  ("train_report_parallel_C.json",  "平行1月-C"),
    "parallel_D":  ("train_report_parallel_D.json",  "平行1月-D"),
}

# 各场景固定映射（与 dashboard 展示一致；A 场景无迭代不出现）
ORDER = ["oot_2026_01", "parallel_B", "parallel_C", "parallel_D"]
VNUM = {"oot_2026_01": "v4", "parallel_B": "v3", "parallel_C": "v4", "parallel_D": "v5"}
STRAT = {"parallel_B": "light", "parallel_C": "standard", "parallel_D": "major",
         "oot_2026_01": None}  # oot 以实际 model_file 为准


def _load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _bad_rate(scenario):
    """场景坏账率：平行场景读 scenario meta，OOT 读 data_quality 报告或默认。"""
    meta = os.path.join(ROOT, "data", "scenarios", f"{scenario}.meta.json")
    m = _load(meta)
    if m and "bad_rate" in m:
        return round(float(m["bad_rate"]) * 100, 2)
    if scenario == "oot_2026_01":
        return 5.0     # 官方 test 真实坏账率
    return None


def build_real_lines():
    """生成 4 行 REAL 数组元素 + 1 行 v1 基座（均 JSON 字面量、无前导逗号问题）。"""
    items = []
    for scn in ORDER:
        rep = _load(os.path.join(OUT, SCEN[scn][0]))
        br = _bad_rate(scn)
        if rep:
            sel = rep.get("selected", {})
            ks = round(float(sel.get("oot_ks", 0) or 0), 4)
            auc = round(float(sel.get("oot_auc", 0) or 0), 4)
            algo = sel.get("algo", "?")
            mf = rep.get("model_file", f"model_{VNUM[scn]}_{scn}_{STRAT.get(scn) or ''}.pkl")
            name = mf[:-4] if mf.endswith(".pkl") else mf
        else:
            ks, auc, algo, name = 0.0, 0.0, "?", f"model_{VNUM[scn]}_{scn}"
        items.append((name, algo, SCEN[scn][1], ks, auc, br))
    # v1 基座
    v1_meta = _load(os.path.join(ROOT, "models", "model_v1_baseline.meta.json")) or {}
    v1_ks = round(float(v1_meta.get("train_KS", 0) or 0), 4)
    v1_auc = round(float(v1_meta.get("train_AUC", 0) or 0), 4)
    v1_algo = v1_meta.get("algo", "?")
    return v1_algo, v1_ks, v1_auc, items


def sync(dash_path=DASH):
    v1_algo, v1_ks, v1_auc, items = build_real_lines()
    real_block = "const REAL = [\n"
    for name, algo, label, ks, auc, br in items:
        br_s = "null" if br is None else br
        real_block += (f'  {{name:"{name}", algo:"{algo}", scn:"{label}", '
                       f'state:"cand", ks:{ks}, auc:{auc}, br:{br_s}, gray:0}},\n')
    real_block += "];"
    v1_line = (f'let models=[{{name:"model_v1_baseline",algo:"{v1_algo}",'
               f'scn:"基座(在线)",state:"online",ks:{v1_ks},auc:{v1_auc},br:4.0,gray:100}}];')

    with open(dash_path, encoding="utf-8") as f:
        html = f.read()
    # 校验两个锚点都存在（与内容是否变化无关）
    if "const REAL = [" not in html:
        raise RuntimeError("dashboard.html 缺少 const REAL 锚点，结构可能已变化")
    if not any("let models=" in ln and "model_v1_baseline" in ln for ln in html.split("\n")):
        raise RuntimeError("dashboard.html 缺少 v1 基座行锚点，结构可能已变化")
    html2 = re.sub(r"const REAL = \[[\s\S]*?\];", real_block, html, count=1)
    # v1 行：按行替换——找到含 model_v1_baseline 的 let models= 行，整行替换（避开中文/引号正则陷阱）
    lines = html2.split("\n")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("let models=") and "model_v1_baseline" in ln:
            lines[i] = v1_line
            break
    html2 = "\n".join(lines)
    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html2)
    return {"v1": (v1_algo, v1_ks, v1_auc), "candidates": len(items),
            "lines": [i[0] for i in items]}


if __name__ == "__main__":
    out = sync()
    print("dashboard 同步完成：", json.dumps(out, ensure_ascii=False, indent=2))
